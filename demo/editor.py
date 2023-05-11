import tkinter as tk

window = tk.Tk()
window.title('GoPilot Demo Editor')

topFrame = tk.Frame()
button = tk.Button(text='Load', master=topFrame)
entry = tk.Entry(master=topFrame)
code = tk.Text(width=100)
label = tk.Label(text='Load a file to start')

button.pack(side='left')
entry.pack(fill=tk.X, side='left', expand=True)
topFrame.pack(fill=tk.X)
code.pack(fill=tk.BOTH, expand=True)
label.pack(side='left')

def handle_button_click(event):
    file_to_load = entry.get()
    if file_to_load.strip() == '':
        label.config(text='Enter a file path', foreground='red')
        return
    try:
        with open(file_to_load, 'r') as f:
            code.delete('1.0', tk.END)
            code.insert('1.0', f.read())
        label.config(text=f'Loaded "{file_to_load}" successfully', foreground='black')
    except:
        label.config(text=f'Error when opening "{file_to_load}"', foreground='red')

button.bind('<Button-1>', handle_button_click)

window.mainloop()