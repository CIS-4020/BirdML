import sys
import cgi
import os
import json
import logging
import threading
import signal
from datetime import datetime

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, parse_qsl, unquote

logging.basicConfig(level=logging.INFO)

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

if __name__ == "__main__":
    httpd = HTTPServer( ( '0.0.0.0', int(sys.argv[1]) ), MyHandler )
    print( "Server listing in port:  ", int(sys.argv[1]) )
    httpd.serve_forever()