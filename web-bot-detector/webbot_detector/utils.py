import time

class WebTrafficFeatureExtractor:
    """Helper class to extract features from web logs/requests"""
    
    @staticmethod
    def extract_features_from_logs(logs):
        """
        Extract features from web server logs
        
        Args:
            logs: List of log entries or log file path
            
        Returns:
            Dictionary containing extracted features
        """
        # This is a simplified implementation
        features = {
            'NUMBER_OF_REQUESTS': len(logs) if isinstance(logs, list) else 0,
            'TOTAL_DURATION': 0,
            'AVERAGE_TIME': 0,
            # ... populate all 30 features ...
        }
        
        return features
        
    @staticmethod
    def extract_features_from_request_session(session_requests):
        """
        Extract features from a session's requests
        
        Args:
            session_requests: List of request data dictionaries
            
        Returns:
            Dictionary with extracted features
        """
        request_count = len(session_requests)
        
        if request_count == 0:
            # Return default values if no requests
            return {feature: 0 for feature in [
                'NUMBER_OF_REQUESTS', 'TOTAL_DURATION', 'AVERAGE_TIME',
                # ... all 30 features ...
            ]}
            
        # Calculate various metrics from session data
        # ... feature extraction logic ...
        
        features = {
            'NUMBER_OF_REQUESTS': request_count,
            # ... populate all features ...
        }
        
        return features