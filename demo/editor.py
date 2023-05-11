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

# label.config(text='sup bro')

window.mainloop()