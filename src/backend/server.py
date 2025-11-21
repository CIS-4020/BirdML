import sys
import cgi
import os
import json
import logging
import threading
import signal
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict
from scripts import class_num_to_name as classConvert

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, parse_qsl, unquote

logging.basicConfig(level=logging.INFO)

from PIL import Image
import io

class MyHandler( BaseHTTPRequestHandler ):

    def handle_one_request(self):
        self.raw_requestline = self.rfile.readline(65537)
        if len(self.raw_requestline) > 65536:
            self.requestline = ''
            self.request_version = ''
            self.command = ''
            self.send_error(414)
            return

        if not self.parse_request():
            return

        # Check the HTTP version
        if self.request_version == "HTTP/2.0":
            self.send_error(505, "HTTP Version Not Supported")
            return

        if self.command == "GET":
            self.do_GET()
        elif self.command == "POST":
            self.do_POST()
        elif self.command == "OPTIONS":
            self.do_OPTIONS()
        else:
            self.send_error(501, "Unsupported method")

    def do_GET(self):

        parsed = urlparse( self.path )

        # give main as default
        if parsed.path in [ '/' ]:

            try:
                self.path = '/frontend/main.html'

                fp = open( '..'+self.path )
                content = fp.read()

                self.send_response( 200 )
                self.send_cors_headers()
                self.send_header( "Content-type", "text/html" )
                self.send_header( "Content-length", len( content ) )
                self.end_headers()

                self.wfile.write( bytes( content, "utf-8" ) )
                fp.close()
            except FileNotFoundError:
                self.send_error(404, "File Not Found: %s" % self.path)
            except Exception as e:
                logging.error(f"Unexpected error: {e}") 

        # give main
        elif parsed.path in [ '/main.html' ]:

            try:
                fp = open( '../frontend'+self.path )
                content = fp.read()

                self.send_response( 200 )
                self.send_cors_headers()
                self.send_header( "Content-type", "text/html" )
                self.send_header( "Content-length", len( content ) )
                self.end_headers()

                self.wfile.write( bytes( content, "utf-8" ) )
                fp.close()
            except FileNotFoundError:
                self.send_error(404, "File Not Found: %s" % self.path)
            except Exception as e:
                logging.error(f"Unexpected error: {e}") 
        
        # give bubblesort
        elif parsed.path in [ '/bubblesort.html' ]:

            try:
                fp = open( '../frontend'+self.path )
                content = fp.read()

                self.send_response( 200 )
                self.send_cors_headers()
                self.send_header( "Content-type", "text/html" )
                self.send_header( "Content-length", len( content ) )
                self.end_headers()

                self.wfile.write( bytes( content, "utf-8" ) )
                fp.close()
            except FileNotFoundError:
                self.send_error(404, "File Not Found: %s" % self.path)
            except Exception as e:
                logging.error(f"Unexpected error: {e}") 

        # give styles
        elif parsed.path in [ '/styles.css' ]:

            try:
                fp = open( '../frontend'+self.path )
                content = fp.read()

                self.send_response( 200 )
                self.send_cors_headers()
                self.send_header( "Content-type", "text/css" )
                self.send_header( "Content-length", len( content ) )
                self.end_headers()

                self.wfile.write( bytes( content, "utf-8" ) )
                fp.close()
            except FileNotFoundError:
                self.send_error(404, "File Not Found: %s" % self.path)
            except Exception as e:
                logging.error(f"Unexpected error: {e}") 

        # give script
        elif parsed.path in [ '/script.js' ]:

            try:
                fp = open( '../frontend'+self.path )
                content = fp.read()

                self.send_response( 200 )
                self.send_cors_headers()
                self.send_header("Content-Type", "application/javascript")
                self.send_header( "Content-length", len( content ) )
                self.end_headers()

                self.wfile.write( bytes( content, "utf-8" ) )
                fp.close()
            except FileNotFoundError:
                self.send_error(404, "File Not Found: %s" % self.path)
            except Exception as e:
                logging.error(f"Unexpected error: {e}") 
    
    def do_POST(self):

        parsed = urlparse( self.path )

        if parsed.path in [ '/sample' ]:

            try:
                
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                arr = data.get('arr', [])
                print("Received array:", arr)

                content = ""

                self.send_response( 200 )
                self.send_cors_headers()
                self.send_header("Content-Type", "text/json")
                self.send_header( "Content-length", len( content ) )
                self.end_headers()

                self.wfile.write( bytes( content, "utf-8" ) )

            except FileNotFoundError:
                self.send_error(404, "File Not Found: %s" % self.path)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

        elif parsed.path in ['/predict']:
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                content_type = self.headers.get('Content-Type')

                if not content_type or "multipart/form-data" not in content_type:
                    self.send_error(400, "Expected multipart/form-data")
                    return

                body = self.rfile.read(content_length)

                boundary = content_type.split("boundary=")[1].encode()
                boundary_bytes = b"--" + boundary

                parts = body.split(boundary_bytes)

                image_bytes = None
                image_name = None

                for part in parts:
                    if b"Content-Disposition" not in part:
                        continue

                    if b'name="image"' in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end == -1:
                            continue

                        image_bytes = part[header_end+4:]
                        if image_bytes.endswith(b"\r\n"):
                            image_bytes = image_bytes[:-2]

                        cd_line = part.split(b"\r\n")[0]
                        if b'filename="' in cd_line:
                            start = cd_line.find(b'filename="') + len(b'filename="')
                            end = cd_line.find(b'"', start)
                            image_name = cd_line[start:end].decode('utf-8')

                        break

                if image_bytes is None:
                    self.send_error(400, "Image not found in form-data")
                    return

                if not image_name:
                    image_name = "uploaded_image"

                print(f"Received image '{image_name}', size: {len(image_bytes)} bytes")

                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert("RGB")

                prediction_result, confidence_score = predict("birdML_50_birds_2.pth", image, image_name)
                print("Prediction result:", prediction_result)

                imagePath = f"../../single_data/{prediction_result}.jpg"
                resultImage = Image.open(imagePath).convert("RGB")

                prediction_string = classConvert.convertClassNumToClassName(prediction_result)

                response = json.dumps({
                    "status": "ok",
                    "image_name": image_name,
                    "size": len(image_bytes),
                    "prediction": prediction_result,
                    "confidence": str(confidence_score),
                    "prediction_string": prediction_string,
                    "prediction_image": pil_to_base64(resultImage)
                })

                self.send_response(200)
                self.send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))

            except Exception as e:
                logging.error(f"Unexpected error in /predict: {e}")
                self.send_error(500, "Server error")


    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

import base64
from io import BytesIO

def pil_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if __name__ == "__main__":
    httpd = HTTPServer( ( '0.0.0.0', int(sys.argv[1]) ), MyHandler )
    print( "Server listing in port:  ", int(sys.argv[1]) )
    httpd.serve_forever()