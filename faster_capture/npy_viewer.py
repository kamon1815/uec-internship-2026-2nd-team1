import base64
import cv2
import tkinter as tk
import numpy as np
import pypuclib
import sys

def npy_viewer(input_file):
    f = open(input_file, 'rb')

    header = np.load(f)
    width, height = header['resolution'].tolist()
    frame_count = header['frame_count'].item()

    decoder = pypuclib.Decoder(header['quantization'].tolist())
    reso = pypuclib.Resolution(width, height)

    frame_start = f.tell()
    np.load(f)
    frame_size = f.tell() - frame_start

    root = tk.Tk()
    root.title('npy_viewer')

    image_label = tk.Label(root)
    image_label.pack()

    frame_index = tk.IntVar(value=0)
    def show_frame(_=None):
        f.seek(frame_start + frame_size * frame_index.get())
        frame = cv2.cvtColor(decoder.decode(np.load(f), reso), cv2.COLOR_GRAY2BGR)
        cv2.putText(
            frame, str(frame_index.get()), (15, 35),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0)
        )
        _, png = cv2.imencode('.png', frame)
        image = tk.PhotoImage(data=base64.b64encode(png).decode())
        image_label.configure(image=image)
        image_label.image = image

    def move(new_index):
        frame_index.set(max(0, min(frame_count - 1, new_index)))
        show_frame()

    jump_value = tk.StringVar(value='0')
    popup = None

    def close_popup():
        nonlocal popup
        if popup is not None:
            popup.destroy()
            popup = None

    def open_popup(_=None):
        nonlocal popup
        if popup is not None:
            close_popup()
            return

        popup = tk.Toplevel(root)
        popup.title('jump_frame')
        popup.transient(root)
        popup.grab_set()
        popup.resizable(False, False)

        validate = popup.register(lambda value: value == '' or value.isdigit())
        entry = tk.Entry(
            popup,
            textvariable=jump_value,
            validate='key',
            validatecommand=(validate, '%P'),
            width=30,
            justify='center'
        )
        entry.pack(padx=20, pady=20)

        jump_value.set(str(frame_index.get()))
        entry.focus_set()
        entry.select_range(0, 'end')

        popup.update_idletasks()
        x = root.winfo_x() + (root.winfo_width() - popup.winfo_width()) // 2
        y = root.winfo_y() + (root.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f'+{x}+{y}')

        def jump(_=None):
            if jump_value.get():
                move(int(jump_value.get()))
            close_popup()

        entry.bind('<Return>', jump)
        popup.bind('<space>', lambda _: close_popup())
        popup.protocol('WM_DELETE_WINDOW', close_popup)
        return 'break'

    controls = tk.Frame(root)
    controls.pack(fill='x')

    scale = tk.Scale(
        controls,
        from_=0,
        to=frame_count - 1,
        orient='horizontal',
        variable=frame_index,
        command=show_frame,
        showvalue=False
    )
    scale.pack(side='left', fill='x', expand=True)

    frame_entry = tk.Entry(
        controls,
        textvariable=frame_index,
        width=10,
        state='readonly'
    )
    frame_entry.pack(side='right')
    frame_entry.bind('<Button-1>', open_popup)

    move_job = None
    move_key = None

    def start_move(n):
        nonlocal move_job, move_key

        if move_key is not None:
            return

        move_key = n
        move(frame_index.get() + n)

        def repeat():
            nonlocal move_job
            move(frame_index.get() + n)
            move_job = root.after(30, repeat)

        move_job = root.after(300, repeat)

    def stop_move(_=None):
        nonlocal move_job, move_key

        if move_job is not None:
            root.after_cancel(move_job)
            move_job = None

        move_key = None

    root.bind('<KeyPress-Left>',  lambda _: start_move(-1))
    root.bind('<KeyPress-Right>', lambda _: start_move(1))
    root.bind('<KeyRelease-Left>', stop_move)
    root.bind('<KeyRelease-Right>', stop_move)

    root.bind('<space>', open_popup)

    def close(_=None):
        root.destroy()
    root.bind('<Escape>', close)

    show_frame()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        f.close()


if __name__ == '__main__':
    #if len(sys.argv) < 2:
    #    print(f'usage: {__file__} <input_file>')
    #    exit(1)

    OUTPUT_FILE =  './faster_capture/output/coin1.npy'

    #npy_viewer(sys.argv[1])
    npy_viewer(OUTPUT_FILE)