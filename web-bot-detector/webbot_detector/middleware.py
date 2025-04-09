from flask import Flask, request, jsonify, Response
import requests
import json
import time
import hashlib
import threading
import logging
from collections import defaultdict
from .utils import WebTrafficFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bot_protection_middleware')

class BotProtectionMiddleware:
    def __init__(self, detection_api_url="http://localhost:5000/detect", 
                 threshold=0.75, cache_duration=300, session_window=15):
        """
        Initialize the middleware
        
        Args:
            detection_api_url: URL of the bot detection service
            threshold: Confidence threshold for bot detection
            cache_duration: How long to cache results (in seconds)
            session_window: Time window to collect session data (in minutes)
        """
        self.detection_api_url = detection_api_url
        self.threshold = threshold
        self.cache_duration = cache_duration
        self.session_window = session_window * 60  # Convert to seconds
        
        # Cache for storing detection results
        self.results_cache = {}
        
        # Session data storage
        self.session_data = defaultdict(list)
        
        # Start a thread to periodically clean old session data
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_sessions, daemon=True)
        self.cleanup_thread.start()
        
        
    # def _extract_features(self, request, session_id):
    #     """Extract features from the request"""
    #     # Basic request data
    #     path = request.path
    #     method = request.method
    #     content_type = request.headers.get('Content-Type', '')
    #     referrer = request.headers.get('Referer', '')
        
    #     # Add request to session data
    #     self.session_data[session_id].append({
    #         'timestamp': time.time(),
    #         'path': path,
    #         'method': method,
    #         'content_type': content_type,
    #         'referrer': referrer
    #     })
        
    #     # Get all requests for this session in the window
    #     session_requests = self.session_data[session_id]
        
    #     # Use the feature extractor from utils.py to process session data
    #     return WebTrafficFeatureExtractor.extract_features_from_request_session(session_requests)

    def _cleanup_old_sessions(self):
        """Periodically clean old session data"""
        while True:
            current_time = time.time()
            keys_to_remove = []
            
            for session_id, requests_data in self.session_data.items():
                # Filter out old requests
                self.session_data[session_id] = [
                    req for req in requests_data 
                    if current_time - req.get('timestamp', 0) <= self.session_window
                ]
                
                # If no requests left, mark session for removal
                if not self.session_data[session_id]:
                    keys_to_remove.append(session_id)
            
            # Remove empty sessions
            for key in keys_to_remove:
                del self.session_data[key]
            
            # Also clean up cache
            cache_keys = list(self.results_cache.keys())
            for key in cache_keys:
                if current_time - self.results_cache[key]['timestamp'] > self.cache_duration:
                    del self.results_cache[key]
            
            # Sleep for a while
            time.sleep(60)

    def _get_session_id(self, request):
        """Extract or generate a session ID from request"""
        # Try to get from cookie
        session_id = request.cookies.get('session_id')
        
        # If not found, use IP + User Agent as fallback
        if not session_id:
            ip = request.remote_addr
            user_agent = request.headers.get('User-Agent', '')
            session_id = hashlib.md5(f"{ip}:{user_agent}".encode()).hexdigest()
        
        return session_id

    def _extract_features(self, request, session_id):
        """Extract features from the request"""
        # Basic request data
        path = request.path
        method = request.method
        content_type = request.headers.get('Content-Type', '')
        referrer = request.headers.get('Referer', '')
        
        # Add request to session data
        self.session_data[session_id].append({
            'timestamp': time.time(),
            'path': path,
            'method': method,
            'content_type': content_type,
            'referrer': referrer
        })
        
        # Get all requests for this session in the window
        session_requests = self.session_data[session_id]
        
        # Count various metrics
        request_count = len(session_requests)
        unique_paths = len(set(req.get('path') for req in session_requests))
        
        # Time between requests
        if len(session_requests) > 1:
            time_diffs = []
            sorted_reqs = sorted(session_requests, key=lambda x: x.get('timestamp', 0))
            for i in range(1, len(sorted_reqs)):
                time_diffs.append(sorted_reqs[i]['timestamp'] - sorted_reqs[i-1]['timestamp'])
            avg_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        else:
            avg_time = 0
        
        # Method counts
        methods = defaultdict(int)
        for req in session_requests:
            methods[req.get('method', '')] += 1
        
        # Create feature vector based on your model's expected input
        features = {
            'NUMBER_OF_REQUESTS': request_count,
            'TOTAL_DURATION': time.time() - session_requests[0]['timestamp'] if request_count > 0 else 0,
            'AVERAGE_TIME': avg_time,
            'STANDARD_DEVIATION': 0,  # Would need more complex calculation
            'REPEATED_REQUESTS': request_count - unique_paths,
            'HTTP_RESPONSE_2XX': 0,  # Would need to track responses
            'HTTP_RESPONSE_3XX': 0,
            'HTTP_RESPONSE_4XX': 0,
            'HTTP_RESPONSE_5XX': 0,
            'GET_METHOD': methods.get('GET', 0),
            'POST_METHOD': methods.get('POST', 0),
            'HEAD_METHOD': methods.get('HEAD', 0),
            'OTHER_METHOD': sum(methods.values()) - methods.get('GET', 0) - methods.get('POST', 0) - methods.get('HEAD', 0),
            'NIGHT': 1 if time.localtime().tm_hour < 6 or time.localtime().tm_hour > 22 else 0,
            'UNASSIGNED': 0,
            'IMAGES': sum(1 for req in session_requests if '.jpg' in req.get('path', '') or '.png' in req.get('path', '')),
            'TOTAL_HTML': sum(1 for req in session_requests if '.html' in req.get('path', '') or 'text/html' in req.get('content_type', '')),
            'HTML_TO_IMAGE': 0,  # More complex calculation needed
            'HTML_TO_CSS': 0,
            'HTML_TO_JS': 0,
            'WIDTH': unique_paths,
            'DEPTH': max([path.count('/') for path in [req.get('path', '') for req in session_requests]]),
            'STD_DEPTH': 0,  # Would need more complex calculation
            'CONSECUTIVE': 0,  # Would need more complex calculation
            'DATA': 0,
            'PPI': 0,
            'SF_REFERRER': sum(1 for req in session_requests if req.get('referrer')),
            'SF_FILETYPE': 0,
            'MAX_BARRAGE': 0,  # Would need more complex calculation
            'PENALTY': 0
        }
        
        return features

    def _check_bot(self, session_id, features):
        """Call the bot detection API"""
        # Check cache first
        if session_id in self.results_cache:
            cache_entry = self.results_cache[session_id]
            if time.time() - cache_entry['timestamp'] <= self.cache_duration:
                logger.debug(f"Cache hit for session {session_id}")
                return cache_entry['result']
        
        # Call the detection API
        try:
            response = requests.post(
                self.detection_api_url,
                json=features,
                headers={'Content-Type': 'application/json'},
                timeout=2
            )
            
            if response.status_code == 200:
                result = response.json()
                # Cache the result
                self.results_cache[session_id] = {
                    'result': result,
                    'timestamp': time.time()
                }
                return result
            else:
                logger.error(f"API error: {response.status_code} {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling bot detection API: {str(e)}")
            return None

    def process_request(self, request):
        """
        Process a request and determine if it's from a bot
        
        Returns:
            dict: Detection result or None if detection failed
        """
        session_id = self._get_session_id(request)
        features = self._extract_features(request, session_id)
        return self._check_bot(session_id, features)


# Example Flask middleware integration
def create_protected_app(bot_detection_url="http://localhost:5000/detect"):
    app = Flask(__name__)
    bot_middleware = BotProtectionMiddleware(detection_api_url=bot_detection_url)
    
    @app.before_request
    def check_for_bot():
        # Skip API endpoints and static files
        if request.path.startswith('/api') or request.path.startswith('/static'):
            return None
            
        print(f"Checking for bot on path: {request}")
        result = bot_middleware.process_request(request)
        
        if result and result.get('is_bot', False) and result.get('confidence', 0) > 0.8:
            # High confidence bot detection - block the request
            return jsonify({
                'error': 'Bot activity detected',
                'code': 'BOT_DETECTED'
            }), 403
        
        # Continue processing the request
        return None
    
    @app.route('/')
    def home():
        return "This is a protected website!"
    
    return app

if __name__ == "__main__":
    # This example shows how to create a protected Flask app
    app = create_protected_app()
    app.run(host="0.0.0.0", port=8080)