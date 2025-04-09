from flask import Flask, request, g, jsonify, redirect, render_template
from webbot_detector.middleware import BotProtectionMiddleware

def create_app(bot_detection_url="http://localhost:5000/detect"):
    app = Flask(__name__)
    bot_middleware = BotProtectionMiddleware(detection_api_url=bot_detection_url)
    
    @app.before_request
    def check_for_bot():
        # Skip API endpoints and static files
        if request.path.startswith('/api') or request.path.startswith('/static'):
            return None
            
        result = bot_middleware.process_request(request)
        g.bot_detection = result
        
        if result and result.get('is_bot', False) and result.get('confidence', 0) > 0.9:
            # High confidence bot detection - block the request
            return jsonify({
                'error': 'Bot activity detected',
                'code': 'BOT_DETECTED'
            }), 403
        
        # Continue processing the request
        return None
    
    @app.route('/')
    def home():
        bot_info = g.bot_detection
        confidence = bot_info.get('confidence', 0) if bot_info else 0
        is_bot = bot_info.get('is_bot', False) if bot_info else False
        
        return f"""
        <html>
            <head><title>Bot Protected Website</title></head>
            <body>
                <h1>Welcome to the bot-protected website!</h1>
                <p>Bot detection: {'Bot detected!' if is_bot else 'Human'} (confidence: {confidence:.2f})</p>
            </body>
        </html>
        """
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)