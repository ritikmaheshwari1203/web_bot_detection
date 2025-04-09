import argparse
import json
from flask import Flask, request, jsonify
from .model import BotDetector

def create_app(detector):
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "ok"})
    
    @app.route('/detect', methods=['POST'])
    def detect_bot():
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            result = detector.predict(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app

def cli():
    parser = argparse.ArgumentParser(description="Web Bot Detection Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the bot detection server")
    server_parser.add_argument("--model-path", required=True, help="Path to the trained model checkpoint")
    server_parser.add_argument("--model-name", default="bert-base-uncased", help="Name of the base model")
    server_parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make a single prediction")
    predict_parser.add_argument("--model-path", required=True, help="Path to the trained model checkpoint")
    predict_parser.add_argument("--model-name", default="bert-base-uncased", help="Name of the base model")
    predict_parser.add_argument("--input", required=True, help="Path to JSON file with input features")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        detector = BotDetector(model_path=args.model_path, model_name=args.model_name)
        app = create_app(detector)
        app.run(host="0.0.0.0", port=args.port)
    
    elif args.command == "predict":
        detector = BotDetector(model_path=args.model_path, model_name=args.model_name)
        with open(args.input, 'r') as f:
            data = json.load(f)
        result = detector.predict(data)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()