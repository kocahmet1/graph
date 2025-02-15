import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import google.generativeai as genai
import os
import ast
import subprocess
import sys
import keyring
import json
from PIL import Image, ImageTk
import base64
import traceback
import time
import glob
import matplotlib

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None
    print("Warning: OpenCV not available - using basic matplotlib rendering")

class GraphGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Graph Generator")
        
        # Configure the main window style
        master.configure(bg='#1a1a2e')  # Dark blue-black background
        
        # Style configuration
        style = ttk.Style()
        style.configure("TFrame", background='#1a1a2e')
        style.configure("TLabel", background='#1a1a2e', foreground='white')
        style.configure("TButton", background='#1a1a2e')

        self.current_description = ""
        self.check_image_after_id = None

        # API Key Section
        self.key_frame = tk.Frame(master, bg='#1a1a2e')
        self.key_label = tk.Label(self.key_frame, text="API Key:", bg='#1a1a2e', fg='white')
        self.key_entry = tk.Entry(self.key_frame, show="*", width=40, bg='white')
        self.save_key_btn = tk.Button(self.key_frame, text="Save Key", command=self.save_api_key,
                                    bg='#2d2d4a', fg='white', activebackground='#3d3d5c', activeforeground='white')
        # Layout
        self.key_label.pack(side=tk.LEFT)
        self.key_entry.pack(side=tk.LEFT, padx=5)
        self.save_key_btn.pack(side=tk.LEFT)
        self.key_frame.pack(pady=5)

        # GUI Components
        self.generate_button = tk.Button(master, text="Generate Graph", state=tk.DISABLED, 
                                       command=self.generate_graph, bg='#2d2d4a', fg='white',
                                       activebackground='#3d3d5c', activeforeground='white')
        self.generate_button.pack(pady=10)

        # Image display panel configuration
        self.image_panel = ttk.Frame(master, style="TFrame")
        self.canvas = tk.Canvas(self.image_panel, bg='white')
        self.scroll_x = ttk.Scrollbar(self.image_panel, orient="horizontal", command=self.canvas.xview)
        self.scroll_y = ttk.Scrollbar(self.image_panel, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.scroll_x.pack(side="bottom", fill="x")
        self.scroll_y.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.image_panel.pack(pady=10, fill="both", expand=True)

        self.image_label = ttk.Label(self.canvas)
        self.image_container = self.canvas.create_window((0,0), window=self.image_label, anchor="nw")

        # Image Upload Section
        self.upload_frame = tk.Frame(master, bg='#1a1a2e')
        button_style = {'bg': '#2d2d4a', 'fg': 'white', 'activebackground': '#3d3d5c', 'activeforeground': 'white'}
        
        self.describe_graph_btn = tk.Button(self.upload_frame, text="Describe Graph", 
                                          command=self.show_graph_description_input, **button_style)
        self.describe_graph_btn.pack(side=tk.LEFT, padx=5)
        
        self.upload_btn = tk.Button(self.upload_frame, text="Upload Image", 
                                  command=self.upload_image, **button_style)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.create_question_btn = tk.Button(self.upload_frame, text="Create New Question", 
                                           command=self.show_question_input, **button_style)
        self.create_question_btn.pack(side=tk.LEFT, padx=5)
        self.upload_frame.pack(pady=5)

        # Question Display
        self.question_frame = tk.Frame(master, bg='#1a1a2e')
        self.question_text = tk.Text(self.question_frame, height=6, wrap=tk.WORD, bg='white')
        self.question_text.pack(fill=tk.BOTH, expand=True)
        self.question_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        # Load saved key
        self.load_api_key()

        # Enable generate button when there's an image
        self.master.after(100, self.check_image)

        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=1)

    def load_api_key(self):
        # Try keyring first
        key = keyring.get_password("graph_generator", "api_key")
        if not key:
            # Fallback to config
            try:
                with open("config.json") as f:
                    config = json.load(f)
                    key = config.get("GOOGLE_API_KEY")
                    if key:
                        self.key_entry.insert(0, key)
            except FileNotFoundError:
                pass

    def save_api_key(self):
        key = self.key_entry.get().strip()
        if not key:
            messagebox.showwarning("Warning", "Please enter an API key")
            return

        # Store in keyring
        try:
            keyring.set_password("graph_generator", "api_key", key)
        except Exception:
            # Fallback to config file
            with open("config.json", "w") as f:
                json.dump({"GOOGLE_API_KEY": key}, f)
        
        messagebox.showinfo("Success", "API key saved securely")

    def generate_graph(self):
        """Generate graph from either text input or stored API description"""
        # Check for stored API description first
        if hasattr(self, 'current_description') and self.current_description.strip():
            self.generate_graph_from_description(self.current_description)
            return
            
        # Fall back to text input if no stored description
        messagebox.showwarning("Warning", "Please process an image first")
        return

    def execute_code_safely(self, code):
        """Execute code in a restricted environment"""
        try:
            print("Starting execute_code_safely...")
            print(f"Current working directory: {os.getcwd()}")
            
            allowed_imports = {'matplotlib': ['pyplot', 'cm'], 'numpy': ['*']}
            
            # Security check
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                    if module.split('.')[0] not in allowed_imports:
                        raise ImportError(f"Import of {module} is not allowed")

            print("Security check passed...")
            
            # Clear any existing plots and close all figures
            plt.close('all')
            print("Cleared existing plots...")
            
            # Create namespace with required imports
            namespace = {
                'plt': plt,
                'np': np,
                'matplotlib': matplotlib
            }
            
            # Remove plt.savefig and plt.close commands from the code
            code_lines = code.split('\n')
            filtered_code = '\n'.join(line for line in code_lines 
                                    if 'plt.savefig' not in line 
                                    and 'plt.close' not in line)
            
            print(f"About to execute code:\n{filtered_code}")
            
            # Execute the plotting code
            exec(filtered_code, namespace)
            print("Code executed successfully...")
            
            # Get the current figure and make sure it's active
            self.figure = plt.gcf()
            plt.figure(self.figure.number)
            
            # Save the figure
            timestamp = int(time.time())
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            output_file = os.path.join(output_dir, f'graph_{timestamp}.png')
            print(f"Will save to: {output_file}")
            
            # Draw the figure before saving
            self.figure.canvas.draw()
            
            # Save with white background
            self.figure.patch.set_facecolor('white')
            for ax in self.figure.get_axes():
                ax.set_facecolor('white')
            
            self.figure.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Saved figure to {output_file}")
            
            # Clear previous figure from canvas if it exists
            for widget in self.canvas.winfo_children():
                widget.destroy()
            print("Cleared previous widgets...")
            
            # Create FigureCanvasTkAgg widget
            canvas_widget = FigureCanvasTkAgg(self.figure, master=self.canvas)
            canvas_widget.draw()
            
            # Get the Tk widget and pack it
            canvas_tk = canvas_widget.get_tk_widget()
            canvas_tk.pack(expand=True, fill='both')
            print("Created and packed canvas widget...")
            
            # Store the current file path
            self.current_file = output_file
            
            # Update canvas scroll region
            self.canvas.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            print("Updated canvas scroll region...")
            
            return True
            
        except Exception as e:
            print(f"Error in execute_code_safely: {str(e)}")
            traceback.print_exc()
            return False

    def display_resized_image(self, path):
        try:
            from PIL import Image, ImageTk
            
            # Clear previous image
            self.image_label.config(image='')
            
            img = Image.open(path)
            max_size = (800, 500)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Delete old graph files except the current one
            for old_file in glob.glob('graph_*.png'):
                if old_file != path:
                    try:
                        os.remove(old_file)
                    except:
                        pass
                        
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to display image: {str(e)}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", ".jpg .jpeg .png")]
        )
        if file_path:
            try:
                self.process_image(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def process_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            genai.configure(api_key=keyring.get_password("graph_generator", "api_key"))
            model = genai.GenerativeModel('gemini-2.0-flash')  # Fixed model name

            response = model.generate_content([
                """Analyze this geometry question image and return a JSON response with EXACTLY this format:
{
    "graph_description": "<matplotlib instructions>",
    "question": "<question text>",
    "answers": ["<answer 1>", "<answer 2>", "..."]
}

DO NOT include any other text, only the JSON object.""",
                {"mime_type": "image/jpeg", "data": image_data}
            ])

            try:
                # First try to parse the response directly
                response_json = json.loads(response.text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown
                try:
                    # Remove markdown code blocks if present
                    clean_response = response.text.replace('```json', '').replace('```', '').strip()
                    response_json = json.loads(clean_response)
                except json.JSONDecodeError as e:
                    # If both attempts fail, show the actual response for debugging
                    error_msg = f"Failed to parse response:\n{response.text}\n\nError: {str(e)}"
                    messagebox.showerror("Error", error_msg)
                    return

            # Display question and answers
            self.question_text.delete("1.0", tk.END)
            self.question_text.insert(tk.END, response_json['question'] + "\n\n")
            for idx, ans in enumerate(response_json['answers'], 1):
                self.question_text.insert(tk.END, f"{idx}. {ans}\n")
            
            # Store description and enable generate button
            self.current_description = response_json.get('graph_description', '')
            if self.current_description:
                self.generate_button.config(state=tk.NORMAL)
            else:
                self.generate_button.config(state=tk.DISABLED)

        except json.JSONDecodeError:
            messagebox.showerror("Error", "Failed to parse API response")

    def process_table_data(self, description):
        try:
            # Extract numeric data from the description
            # Example format for the electric company data:
            cell_data = [
                [48, 623, 671],
                [130, 90, 220],
                [178, 713, 891]
            ]
            
            col_labels = ['Asked for\nrepairs', 'Did not ask\nfor repairs', 'Total']
            row_labels = ['Asked\nabout a bill', 'Did not ask\nabout a bill', 'Total']
            
            return {
                'cell_data': cell_data,
                'col_labels': col_labels,
                'row_labels': row_labels
            }
            
        except Exception as e:
            print(f"Error processing table data: {str(e)}")
            return None

    def generate_graph_from_description(self, description):
        if not description.strip():
            raise ValueError("Empty description")
        
        try:
            # Get API key
            key = keyring.get_password("graph_generator", "api_key")
            if not key:
                messagebox.showerror("Error", "No API key found. Please save your key first.")
                return

            # Configure Gemini
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.0-flash')  # Fixed model name

            # Generate code with precise specifications
            prompt_parts = f'''Generate precise matplotlib code that exactly matches this graph description:
{description}

Requirements:
1. First determine the graph type (table, bar chart, line chart, geometric shape)
2. Use appropriate plotting function:
   - Tables: plt.table() with cellLoc='center'
   - Charts: plt.bar()/plt.plot() with proper styling
   - Shapes: plt.Polygon or plt.Circle
3. Set figure size to be compact:
   - For tables: plt.figure(figsize=(4, 3))
   - For charts: plt.figure(figsize=(5, 4))
   - For shapes: plt.figure(figsize=(4, 3))
4. Include plt.tight_layout() for non-table layouts
5. Save with bbox_inches='tight' and adequate DPI

Example formats:
```python
# Table example
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3))
table = plt.table(cellText=[[...]],
                rowLabels=[...],
                colLabels=[...],
                loc='center',
                cellLoc='center')
plt.axis('off')

# Line chart example
plt.figure(figsize=(5, 4))
plt.plot([1,2,3], [4,5,6], marker='o', linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.grid(True)
plt.tight_layout()

# Bar chart example
plt.figure(figsize=(5, 4))
plt.bar(['A','B','C'], [10,20,15], color='skyblue')
plt.xlabel('...')
plt.ylabel('...')
plt.xticks(rotation=45)
plt.tight_layout()
```
'''
            response = model.generate_content(
                prompt_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2000,
                )
            )

            print('\n=== FULL AI RESPONSE ===')
            print(response.text)
            print('=== END FULL AI RESPONSE ===\n')
            
            try:
                generated_code = response.text.split('```python')[1].split('```')[0].strip()
                print('\n=== EXTRACTED CODE ===')
                print(generated_code)
                print('\n=== END OF EXTRACTED CODE ===')
                
                # Execute with better error handling
                print(f"About to execute generated code...")
                result = self.execute_code_safely(generated_code)
                print(f"Code execution result: {result}")
                
            except Exception as e:
                print(f"Error in generate_graph_from_description: {str(e)}")
                print(f"Full traceback: {traceback.format_exc()}")
                messagebox.showerror("Execution Error", f"Graph generation failed:\n{str(e)}")
                return

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate graph: {str(e)}")

    def create_table(self, data):
        # Clear any existing plots
        plt.clf()
        
        # Extract table data
        try:
            # Convert all data to strings to avoid type issues
            cell_text = [[str(cell) for cell in row] for row in data['cell_data']]
            col_labels = [str(label) for label in data['col_labels']]
            row_labels = [str(label) for label in data['row_labels']]
            
            # Create figure and axis
            plt.figure(figsize=(8, 6))  # Increased figure size
            plt.table(cellText=cell_text,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center', 
                      cellLoc='center',
                      colWidths=[0.2]*3)
            plt.tight_layout()  # Add this line
            plt.savefig('temp_graph.png', bbox_inches='tight', dpi=300)  # Ensure tight bounding box
            
            self.figure = plt.gcf()
            
            # Clear previous figure from canvas if it exists
            for widget in self.canvas.winfo_children():
                widget.destroy()
            
            # Create FigureCanvasTkAgg widget
            canvas = FigureCanvasTkAgg(self.figure, master=self.canvas)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # Draw the canvas
            canvas.draw()
            return True
            
        except Exception as e:
            print(f"Error creating table: {str(e)}")
            return False

    def generate_from_description(self):
        if self.current_description.strip():
            self.generate_graph_from_description(self.current_description)
        else:
            messagebox.showwarning("No Description", "Process an image first")

    def check_image(self):
        """Check if an image is loaded and enable/disable generate button accordingly"""
        if hasattr(self, 'current_description') and self.current_description.strip():
            self.generate_button.config(state=tk.NORMAL)
        else:
            self.generate_button.config(state=tk.DISABLED)
            
        # Cancel any existing after callback
        if self.check_image_after_id:
            self.master.after_cancel(self.check_image_after_id)
        
        # Schedule next check
        self.check_image_after_id = self.master.after(100, self.check_image)

    def _enhance_details(self, image_path):
        if cv2 is None or np is None:
            return
        try:
            img = cv2.imread(image_path)
            if img is None:
                return
            laplacian = cv2.Laplacian(img, cv2.CV_16S)
            sharpened = cv2.convertScaleAbs(cv2.addWeighted(img, 1.5, laplacian, -0.5, 0))
            cv2.imwrite(image_path, sharpened)
        except Exception as e:
            print(f"Image enhancement failed: {str(e)}")

    def save_graph(self, filename):
        self.figure.savefig(filename, dpi=300, bbox_inches='tight')
        self._enhance_details(filename)  # Apply OpenCV enhancement
        self.current_file = filename

    def show_question_input(self):
        """Show the question input dialog"""
        # Create popup dialog
        dialog = tk.Toplevel(self.master)
        dialog.title("Create New Question")
        dialog.geometry("400x250")
        dialog.transient(self.master)  # Make dialog modal
        dialog.grab_set()  # Make dialog modal
        
        # Center the dialog on the screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Add padding around the frame
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Question input label
        label = tk.Label(main_frame, text="Enter sample question:", font=("Arial", 10))
        label.pack(anchor="w", pady=(0, 5))

        # Question input text area
        question_input = tk.Text(main_frame, height=5, wrap=tk.WORD)
        question_input.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Generate button
        def generate():
            sample_question = question_input.get(1.0, tk.END).strip()
            dialog.destroy()
            if sample_question:
                self.generate_question_and_graph(sample_question)

        generate_btn = tk.Button(main_frame, text="Generate Question & Graph", command=generate)
        generate_btn.pack(pady=(0, 10))

        # Cancel button
        cancel_btn = tk.Button(main_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack()

        # Set focus to text area
        question_input.focus_set()

    def generate_question_and_graph(self, sample_question):
        """Generate a question and graph based on user input"""
        if not sample_question:
            messagebox.showwarning("Warning", "Please enter a sample question")
            return

        key = self.key_entry.get().strip()
        if not key:
            messagebox.showwarning("Warning", "Please enter your API key")
            return

        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""Given this sample question: "{sample_question}"

            Create a mathematical or statistical question with a graph that helps answer it. Follow these requirements:

            1. First determine the most appropriate graph type:
               - Table: For comparing multiple values or showing data in a grid
               - Bar chart: For comparing quantities across categories
               - Line chart: For showing trends over time or continuous relationships
               - Geometric shape: For geometry or spatial relationship questions

            2. Create a clear question text and 4 multiple choice answers (A, B, C, D) that relate directly to the graph.

            3. Generate Python code that follows these exact specifications:
               - For tables:
                 ```python
                 plt.figure(figsize=(4, 3))
                 table = plt.table(cellText=[[...]],
                                 rowLabels=[...],
                                 colLabels=[...],
                                 loc='center',
                                 cellLoc='center')
                 plt.axis('off')
                 ```
               
               - For line charts:
                 ```python
                 plt.figure(figsize=(5, 4))
                 plt.plot([...], [...], marker='o')  # Add appropriate styling
                 plt.xlabel('...')
                 plt.ylabel('...')
                 plt.grid(True)
                 plt.tight_layout()
                 ```
               
               - For bar charts:
                 ```python
                 plt.figure(figsize=(5, 4))
                 plt.bar([...], [...], color='skyblue')  # Use clear colors
                 plt.xlabel('...')
                 plt.ylabel('...')
                 plt.xticks(rotation=45)
                 plt.tight_layout()
                 ```

               - For geometric shapes:
                 ```python
                 plt.figure(figsize=(4, 3))
                 # Use plt.Polygon() or plt.Circle() as needed
                 plt.axis('equal')
                 plt.grid(True)
                 plt.tight_layout()
                 ```

            Format your response exactly like this:
            QUESTION: [question text]
            CHOICES:
            A) [choice text]
            B) [choice text]
            C) [choice text]
            D) [choice text]
            GRAPH_CODE:
            ```python
            [matplotlib code following the specifications above]
            ```"""

            response = model.generate_content(prompt)
            response_text = response.text

            # Parse the response
            parts = {}
            current_section = None
            code_block = []
            in_code_block = False

            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('QUESTION:'):
                    current_section = 'question'
                    parts[current_section] = line[9:].strip()
                elif line.startswith('CHOICES:'):
                    current_section = 'choices'
                    parts[current_section] = []
                elif line.startswith(('A)', 'B)', 'C)', 'D)')) and current_section == 'choices':
                    parts[current_section].append(line)
                elif line.startswith('GRAPH_CODE:'):
                    current_section = 'code'
                    continue
                elif line.startswith('```python'):
                    in_code_block = True
                    continue
                elif line.startswith('```'):
                    in_code_block = False
                    parts['code'] = '\n'.join(code_block)
                elif in_code_block:
                    code_block.append(line)

            # Display question and choices
            question_text = f"{parts['question']}\n\n"
            for choice in parts['choices']:
                question_text += f"{choice}\n"
            
            self.question_text.delete(1.0, tk.END)
            self.question_text.insert(1.0, question_text)

            # Generate the graph
            self.execute_code_safely(parts['code'])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate question and graph: {str(e)}")
            traceback.print_exc()

    def show_graph_description_input(self):
        """Show the graph description input dialog"""
        # Create popup dialog
        dialog = tk.Toplevel(self.master)
        dialog.title("Create Graph from Description")
        dialog.geometry("500x300")
        dialog.transient(self.master)  # Make dialog modal
        dialog.grab_set()  # Make dialog modal
        
        # Center the dialog on the screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Add padding around the frame
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Description input label with example
        label = tk.Label(main_frame, text="Describe the graph you want to create:", font=("Arial", 10, "bold"))
        label.pack(anchor="w", pady=(0, 5))
        
        example_text = "Example: 'Create a line graph showing temperature variations over 24 hours,\nstarting at 15°C at midnight, reaching 25°C at noon, and ending at 18°C.'"
        example_label = tk.Label(main_frame, text=example_text, font=("Arial", 9), justify=tk.LEFT, fg="gray")
        example_label.pack(anchor="w", pady=(0, 10))

        # Description input text area
        description_input = tk.Text(main_frame, height=6, wrap=tk.WORD)
        description_input.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Generate button
        def generate():
            description = description_input.get(1.0, tk.END).strip()
            dialog.destroy()
            if description:
                self.generate_graph_from_description(description)

        generate_btn = tk.Button(main_frame, text="Generate Graph", command=generate)
        generate_btn.pack(pady=(0, 10))

        # Cancel button
        cancel_btn = tk.Button(main_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack()

        # Set focus to text area
        description_input.focus_set()

    def generate_graph_from_description(self, description):
        """Generate a graph based on the user's description"""
        if not description:
            messagebox.showwarning("Warning", "Please enter a graph description")
            return

        key = self.key_entry.get().strip()
        if not key:
            messagebox.showwarning("Warning", "Please enter your API key")
            return

        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""Create Python code using matplotlib to generate the following graph:
            "{description}"

            Requirements:
            1. Use appropriate graph type based on the data and description
            2. Follow these specifications:
               - For tables:
                 ```python
                 plt.figure(figsize=(4, 3))
                 table = plt.table(cellText=[[...]], 
                                 rowLabels=[...],
                                 colLabels=[...],
                                 loc='center',
                                 cellLoc='center')
                 plt.axis('off')
                 ```
               
               - For line charts:
                 ```python
                 plt.figure(figsize=(5, 4))
                 plt.plot([...], [...], marker='o')  # Add appropriate styling
                 plt.xlabel('...')
                 plt.ylabel('...')
                 plt.grid(True)
                 plt.tight_layout()
                 ```
               
               - For bar charts:
                 ```python
                 plt.figure(figsize=(5, 4))
                 plt.bar([...], [...], color='skyblue')  # Use clear colors
                 plt.xlabel('...')
                 plt.ylabel('...')
                 plt.xticks(rotation=45)
                 plt.tight_layout()
                 ```

               - For geometric shapes:
                 ```python
                 plt.figure(figsize=(4, 3))
                 # Use plt.Polygon() or plt.Circle() as needed
                 plt.axis('equal')
                 plt.grid(True)
                 plt.tight_layout()
                 ```

            3. Include proper labels, titles, and grid where appropriate
            4. Use clear, readable fonts and colors
            5. Apply tight_layout() for proper spacing

            Return ONLY the Python code without any explanation or markdown formatting.
            The code must start with 'import matplotlib.pyplot as plt' and contain no text before this import.
            Do not include markdown code block markers (```python or ```)."""

            response = model.generate_content(prompt)
            code = response.text.strip()
            
            # Remove markdown code blocks if present
            if "```" in code:
                # Split by ``` and take the middle part if it exists
                parts = code.split("```")
                if len(parts) >= 3:
                    code = parts[1]
                    if code.startswith("python"):
                        code = code[6:].strip()
                else:
                    # If we can't find a proper code block, just remove the ``` markers
                    code = code.replace("```", "").strip()
            
            # Clean up the code
            lines = code.splitlines()
            cleaned_lines = []
            started = False
            
            for line in lines:
                line = line.strip()
                # Skip empty lines before the import
                if not started and not line:
                    continue
                # Start collecting lines from the matplotlib import
                if line.startswith("import matplotlib"):
                    started = True
                if started:
                    cleaned_lines.append(line)
            
            # If no import statement was found, add it
            if not cleaned_lines or not cleaned_lines[0].startswith("import matplotlib"):
                cleaned_lines.insert(0, "import matplotlib.pyplot as plt")
            
            # Join the lines back together
            code = "\n".join(cleaned_lines)
            
            # Execute the code to generate the graph
            self.execute_code_safely(code)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate graph: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphGeneratorApp(root)
    root.mainloop()
