import bottle
import ngrok
import pathlib
import threading

BASE_DIR = pathlib.Path(__file__).resolve().parent
bottle.TEMPLATE_PATH.append(BASE_DIR / 'views')

HOST = 'localhost'
PORT = 8080
PUBLIC_URL = 'cleaver-fraction-art.ngrok-free.dev'  # https://cleaver-fraction-art.ngrok-free.dev


@bottle.get('/')
def index():
    return bottle.template('index')


def run_server():
    bottle.run(host=HOST, port=PORT)

def main():
    ngrok.forward(f'{HOST}:{PORT}', authtoken_from_env=True, domain=PUBLIC_URL)
    threading.Thread(target=run_server, daemon=True).start()

    try:
        while True:
            pass

    except KeyboardInterrupt:
        pass
    
if __name__ == '__main__':
    main()
