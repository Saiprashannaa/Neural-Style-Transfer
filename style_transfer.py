import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import time
from threading import Timer

class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Style Transfer")
        self.root.geometry("1100x700")
        self.root.configure(bg="#252525")

        self.content_image = None
        self.style_image = None
        self.stylized_image = None
        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        
        self.resize_timer = None

        self.create_widgets()
        self.root.bind("<Configure>", self.on_resize)

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=6)
        style.configure("TLabel", background="#252525", foreground="white", font=("Arial", 12))

        main_frame = tk.Frame(self.root, bg="#252525")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        control_frame = tk.Frame(main_frame, bg="#353535", pady=10, padx=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        ttk.Button(control_frame, text="📂 Content Image", command=lambda: self.load_image("content")).pack(side=tk.LEFT, padx=15)
        ttk.Button(control_frame, text="🎨 Style Image", command=lambda: self.load_image("style")).pack(side=tk.LEFT, padx=15)
        ttk.Button(control_frame, text="✨ Apply Style", command=self.process_image).pack(side=tk.LEFT, padx=15)
        ttk.Button(control_frame, text="💾 Save Image", command=self.save_image).pack(side=tk.LEFT, padx=15)

        image_frame = tk.Frame(main_frame, bg="#252525")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.content_label_frame = tk.LabelFrame(image_frame, text="Content Image", bg="#252525", fg="white", font=("Arial", 12))
        self.content_label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.content_label_frame.pack_propagate(False)
        self.content_label = tk.Label(self.content_label_frame, bg="#2C2C2C", relief=tk.FLAT)
        self.content_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.style_label_frame = tk.LabelFrame(image_frame, text="Style Image", bg="#252525", fg="white", font=("Arial", 12))
        self.style_label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.style_label_frame.pack_propagate(False)
        self.style_label = tk.Label(self.style_label_frame, bg="#2C2C2C", relief=tk.FLAT)
        self.style_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_label_frame = tk.LabelFrame(image_frame, text="Stylized Image", bg="#252525", fg="white", font=("Arial", 12))
        self.result_label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_label_frame.pack_propagate(False)
        self.result_label = tk.Label(self.result_label_frame, bg="#2C2C2C", relief=tk.FLAT)
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def on_resize(self, event):
        if self.resize_timer:
            self.resize_timer.cancel()
        self.resize_timer = Timer(0.2, self.update_displayed_images)
        self.resize_timer.start()

    def update_displayed_images(self):
        if self.content_image:
            self.show_image(self.content_image, self.content_label)
        if self.style_image:
            self.show_image(self.style_image, self.style_label)
        if self.stylized_image:
            self.show_image(self.stylized_image, self.result_label)

    def load_image(self, image_type):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        try:
            img = Image.open(file_path)
            if image_type == "content":
                self.content_image = img
                self.show_image(img, self.content_label)
            else:
                self.style_image = img
                self.show_image(img, self.style_label)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def process_image(self):
        if not self.content_image:
            messagebox.showwarning("Warning", "Please select a content image first!")
            return
        if not self.style_image:
            messagebox.showwarning("Warning", "Please select a style image first!")
            return
        
        try:
            content_tensor = self.preprocess_image(np.array(self.content_image))
            style_tensor = self.preprocess_image(np.array(self.style_image))
            
            outputs = self.model(tf.constant(content_tensor), tf.constant(style_tensor))
            stylized_image = outputs[0]
            
            result_img = self.postprocess_image(stylized_image)
            self.stylized_image = result_img
            self.show_image(result_img, self.result_label)
        except Exception as e:
            messagebox.showerror("Error", f"Style transfer failed:\n{str(e)}")
    
    def save_image(self):
        if self.stylized_image is None:
            messagebox.showerror("Error", "No stylized image to save. Process an image first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                self.stylized_image.save(file_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def preprocess_image(self, image):
        img = tf.image.convert_image_dtype(image, tf.float32)
        img = tf.image.resize(img, (512, 512))
        img = img[tf.newaxis, :]
        return img

    def postprocess_image(self, image):
        img = image.numpy()[0]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def show_image(self, image, label_widget):
        frame_size = min(label_widget.winfo_width(), label_widget.winfo_height())
        if frame_size > 0:
            img = image.resize((frame_size, frame_size), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.configure(image=photo)
            label_widget.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferApp(root)
    root.mainloop()
