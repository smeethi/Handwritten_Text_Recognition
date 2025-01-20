import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import main
from pyngrok import ngrok
import threading

def recognize_image():
    filepath = image_path.get()
    recognized_text = main.recognize_handwritten_image(filepath)
    output_textbox.delete("1.0", tk.END)
    output_textbox.insert(tk.END, recognized_text)

def open_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if filepath:
        image_path.set(filepath)
        image = Image.open(filepath)
        image.thumbnail((400, 400))
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image

def start_ngrok():
    ngrok_tunnel = ngrok.connect(8888)
    print("Ngrok Tunnel:", ngrok_tunnel)

# Create the main window
window = tk.Tk()
window.title("Handwritten Image Recognition")

# Create a variable to store the image path
image_path = tk.StringVar()

# Create the image upload button
upload_button = tk.Button(window, text="Upload Image", command=open_image)
upload_button.pack(pady=10)

# Create the image display label
image_label = tk.Label(window)
image_label.pack()

# Create the recognize button
recognize_button = tk.Button(window, text="Recognize Image", command=recognize_image)
recognize_button.pack(pady=10)

# Create the output text box
output_textbox = tk.Text(window, height=10, width=50)
output_textbox.pack(pady=10)

# Start the GUI event loop
threading.Thread(target=start_ngrok).start()
window.mainloop()