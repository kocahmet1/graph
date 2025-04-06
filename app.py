from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import base64
import json
import os
import ast
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

class WebGraphGenerator:
    def __init__(self):
        self.current_description = ""
        self.setup_api()
        self.model_preference = "slow"  # Default to slow mode
        self.api_key = None

    def setup_api(self):
        try:
            try:
                with open("config.json") as f:
                    config = json.load(f)
                    api_key = config.get("GOOGLE_API_KEY")
                    if not api_key:
                        print("No API key found in config.json")
                        return False
                    
                    # Store the API key
                    self.api_key = api_key
                    
                    # Load model preference from config if available
                    self.model_preference = config.get("MODEL_PREFERENCE", "slow")
            except FileNotFoundError:
                print("config.json file not found")
                return False
            except json.JSONDecodeError:
                print("Invalid JSON in config.json file")
                return False
            except Exception as e:
                print(f"Error reading config.json: {str(e)}")
                return False
                
            # Now let's try to configure the genai library
            try:
                # Configure the genai library
                genai.configure(api_key=api_key)
                
                # Test the API key with a small request
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content("Test")
                
                # If we got here, the API key is valid
                return True
            except Exception as api_error:
                print(f"Error configuring Gemini API: {str(api_error)}")
                return False
        except Exception as e:
            print(f"Error setting up API: {str(e)}")
            return False
            
    def update_api_key(self, new_api_key):
        """Update the API key in config.json and reconfigure genai"""
        try:
            # Make sure we have a valid API key string
            if not new_api_key or not isinstance(new_api_key, str) or len(new_api_key) < 10:
                return False, "Invalid API key format"
            
            # Save the old API key in case we need to revert
            old_api_key = self.api_key
            
            # First test that the API key is valid
            try:
                genai.configure(api_key=new_api_key)
                
                # Try to initialize a model with the new API key to verify it works
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content("Test")
                
                # If we get here, the API key is valid
                # Update the stored API key
                self.api_key = new_api_key
            except Exception as e:
                # API key validation failed
                # Revert to previous key if there was one
                if old_api_key:
                    genai.configure(api_key=old_api_key)
                return False, f"Invalid API key: {str(e)}"
            
            # Now save to config.json
            try:
                # First read the existing config
                try:
                    with open("config.json", "r") as f:
                        config = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    # If the file doesn't exist or is invalid, create a new config
                    config = {}
                
                # Update the API key in the config
                config["GOOGLE_API_KEY"] = new_api_key
                
                # Save the updated config
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=4)
                
                return True, None
            except Exception as file_error:
                # If we can't save to the file, still keep using the new API key in memory
                # but report the error
                return False, f"API key is valid but couldn't save to config.json: {str(file_error)}"
        except Exception as e:
            return False, f"Error updating API key: {str(e)}"

    def get_model(self, task_type="default"):
        """
        Returns the appropriate model based on current preference and task type
        task_type: Can be 'default', 'image', 'code', or other task-specific identifiers
        """
        if self.model_preference == "fast":
            # Fast mode: use gemini-2.0-flash for everything
            return genai.GenerativeModel('gemini-2.0-flash')
        else:
            # Slow mode: use the more capable model for complex tasks
            if task_type in ["default", "code", "graph_generation"]:
                return genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
            else:
                # For simpler tasks, still use the faster model
                return genai.GenerativeModel('gemini-2.0-flash')

    def save_model_preference(self):
        """Save the current model preference to config.json"""
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            
            config["MODEL_PREFERENCE"] = self.model_preference
            
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving model preference: {str(e)}")
            return False

    def generate_graph_from_description(self, description):
        try:
            model = self.get_model(task_type="graph_generation")
            prompt = f'''Generate precise matplotlib code that will EXACTLY match the graph description with perfect accuracy:
"{description}"

Requirements:
1. First perform detailed analysis:
   - Extract ONLY graph-related information from description
   - EXCLUDE all question text and answer choices
   - Focus ONLY on visual elements and labels that are part of the graph
   - Identify PRECISE data points and relationships
   - Record EXACT dimensions, scales, and proportions

2. Content filtering:
   - Include ONLY:
     * Graph elements (lines, bars, shapes)
     * Axis labels and scales
     * Data point labels
     * Graph-specific annotations
     * Legend if required
   - EXCLUDE:
     * Question text
     * Answer choices
     * Multiple choice options
     * Any text not directly part of the graph
     * Explanatory text outside the graph

3. Use appropriate visualization with EXACT specifications:
   For data comparison charts:
   - Bar charts:
     * Use EXACT heights for each bar
     * Maintain PRECISE bar widths and spacing
     * Position labels at EXACT coordinates
     * Example: plt.bar(x, heights, width=0.8, align='center')

   - Line/Scatter plots:
     * Plot EXACT data point coordinates
     * Ensure intersections occur at PRECISE points
     * Use exact numerical values:
       x1, y1 = [0, 20, 40], [0, 30, 60]  # First line
       x2, y2 = [0, 20, 40], [60, 30, 0]  # Second line
     * High-resolution plotting: plt.plot(x, y, '-', linewidth=1.5)

   For geometric shapes:
   - CRITICAL: Preserve EXACT vertex-label associations
     * If angle B is right angle, it MUST be at vertex B
     * If point P is at origin, label P MUST be at (0,0)
     * Match labels to vertices based on geometric properties

   - Use mathematical properties to determine label placement:
     * Right angles: Place label at the right angle vertex
     * Equal sides: Labels must match equal side endpoints
     * Special points: Origin, centroid, focus points must match labels

   - Implement precise label positioning:
     plt.text(x, y, 'B', 
             horizontalalignment='right' if x < 0 else 'left',
             verticalalignment='top' if y < 0 else 'bottom',
             transform=ax.transData,
             offset_points=(5, 5))  # Fine-tune offset

   - For specific geometric shapes:
     * Triangles: 
       - Right triangles: Right angle label MUST match description
       - Isosceles: Equal side vertex labels must match
       - Equilateral: Maintain 60° angle label positions
     * Circles:
       - Center label placement must match description
       - Radius labels at exact endpoints
     * Polygons:
       - Regular: Maintain symmetric label positions
       - Irregular: Match label positions to vertex properties

   - Verify geometric properties:
     * Check angle types match labels (right, acute, obtuse)
     * Verify parallel/perpendicular sides have correct labels
     * Confirm special points (centroids, foci) match labels
     * Ensure equal parts have corresponding labels

   For statistical visualizations:
   - Histograms: Use exact bin edges and counts
   - Box plots: Show precise quartile positions
   - Pie charts: Use exact percentage values
   - Error bars: Show exact uncertainty ranges

4. Implement precise positioning:
   - Labels and annotations:
     plt.annotate('Label', xy=(x, y), xytext=(0, 0),
                 textcoords='offset points',
                 ha='center', va='center')
   - Axis control:
     * Set exact limits: plt.xlim(xmin, xmax)
     * Use precise ticks: plt.xticks([exact_positions])
     * Maintain proper aspect ratio when needed
   - Legend placement:
     * Use exact positioning: plt.legend(loc='upper left', bbox_to_anchor=(x, y))

5. Ensure data accuracy:
   - Maintain exact numerical precision
   - Calculate precise:
     * Values and measurements
     * Proportions and ratios
     * Intervals and gaps
     * Statistical measures
   - Use exact scales and ranges
   - Preserve data relationships

6. Quality control checks:
   - Verify ONLY graph elements are included
   - Confirm NO question text or answers are present
   - Validate ALL labels and annotations are graph-related
   - Check ALL relationships and proportions
   - Ensure proper scaling across ALL dimensions
   - VERIFY label-vertex associations match description

Generate matplotlib code that reproduces ONLY the graph with perfect accuracy.
Focus on EXACT numerical values and precise positioning of graph elements ONLY.'''

            response = model.generate_content(prompt)
            text = response.text.strip()

            # Try to extract code from markdown if present
            if "```python" in text:
                code = text.split("```python")[1].split("```")[0].strip()
            elif "```" in text:
                code = text.split("```")[1].split("```")[0].strip()
            else:
                # If no markdown, try to find just the Python code
                import re
                # Look for typical Python patterns
                python_patterns = [
                    r'import matplotlib\.pyplot',
                    r'plt\.',
                    r'import numpy',
                    r'np\.'
                ]
                if any(re.search(pattern, text) for pattern in python_patterns):
                    code = text
                else:
                    raise ValueError("Could not extract valid Python code from response")

            # Execute the code and get the plot
            plt.clf()
            namespace = {'plt': plt, 'np': np}
            exec(code, namespace)

            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            plt.close()

            return buf

        except Exception as e:
            print(f"Error generating graph: {str(e)}")
            return None

    def extract_question_info(self, image_data):
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Use a more capable model for complex text extraction
            model = self.get_model(task_type="image")
            prompt = """Analyze this image of a reading comprehension or test question and extract ALL of the following components if present:

1. Any PASSAGE text (reading material, context, story, or information that precedes the question)
2. The COMPLETE question text
3. All answer choices (A, B, C, D, etc.)

Format your response as follows:
PASSAGE: [complete passage text if present, otherwise "No passage"]
QUESTION: [full question text]
CHOICES:
A. [complete first choice text]
B. [complete second choice text]
C. [complete third choice text]
D. [complete fourth choice text]

VERY IMPORTANT INSTRUCTIONS:
- INCLUDE the ENTIRE passage and ALL text in the image
- Extract ALL text in the image, even if it spans multiple paragraphs
- Maintain paragraph breaks in the passage text
- Do NOT omit or truncate ANY content from the passage
- Include ALL formulas, symbols, and special characters exactly as shown
- Preserve mathematical notations, equations, and formatting as much as possible
- If tables, graphs or diagrams are described in text, include those descriptions
- For each answer choice, present it as ONE COMPLETE STATEMENT
- Do NOT include bullet points (•) at the beginning of answer choices
- Do NOT include square brackets around answer choices
- Do NOT split a single answer choice into multiple parts
- Put each answer choice on its own line, preceded by its letter (A, B, C, D)

Just provide the extracted text in this format, with no additional analysis or explanation."""

            # Convert PIL Image to the format expected by Gemini
            response = model.generate_content([{"mime_type": "image/png", "data": image_data}, prompt])
            response_text = response.text.strip()

            # Parse the response to extract passage, question and answers
            question_info = {"passage": "", "question": "", "answers": []}
            
            # Check for the different sections
            passage_present = "PASSAGE:" in response_text
            question_present = "QUESTION:" in response_text
            choices_present = "CHOICES:" in response_text
            
            if passage_present and question_present and choices_present:
                # Split the text by the major sections
                passage_parts = response_text.split("QUESTION:")
                passage_text = passage_parts[0].replace("PASSAGE:", "").strip()
                
                remaining_text = passage_parts[1]
                question_parts = remaining_text.split("CHOICES:")
                question_text = question_parts[0].strip()
                choices_text = question_parts[1].strip() if len(question_parts) > 1 else ""
                
                # Handle case where passage contains "No passage"
                if passage_text.lower() == "no passage":
                    question_info["passage"] = ""
                else:
                    question_info["passage"] = passage_text
                
                question_info["question"] = question_text
                # Split by each letter choice on a new line rather than by commas
                import re
                answer_pattern = re.compile(r'([A-D]\..+?)(?=[A-D]\.|$)', re.DOTALL)
                matches = answer_pattern.findall(choices_text + "\n")
                if matches:
                    question_info["answers"] = [match.strip() for match in matches]
                else:
                    # Fallback to old method if regex doesn't match anything
                    question_info["answers"] = [choice.strip() for choice in choices_text.split(',')]
            else:
                # Fallback to line-by-line parsing
                lines = response_text.split('\n')
                passage_lines = []
                question_lines = []
                in_passage = False
                in_question = False
                
                for line in lines:
                    if line.startswith("PASSAGE:"):
                        in_passage = True
                        in_question = False
                        passage_line = line.replace("PASSAGE:", "").strip()
                        if passage_line and passage_line.lower() != "no passage":
                            passage_lines.append(passage_line)
                    elif line.startswith("QUESTION:"):
                        in_passage = False
                        in_question = True
                        question_lines.append(line.replace("QUESTION:", "").strip())
                    elif line.startswith("CHOICES:"):
                        in_passage = False
                        in_question = False
                        choices_text = line.replace("CHOICES:", "").strip()
                        
                        # Extract answer choices by letter (A, B, C, D) format
                        answer_list = []
                        lines = choices_text.strip().split('\n')
                        
                        # Try to detect answer format - first check if they're on separate lines
                        if len(lines) > 1 and any(re.match(r'^[A-D]\. ', line.strip()) for line in lines):
                            current_answer = ""
                            current_letter = ""
                            
                            for answer_line in lines:
                                # If this line starts a new answer
                                if re.match(r'^[A-D]\. ', answer_line.strip()):
                                    # Save the previous answer if there was one
                                    if current_answer:
                                        answer_list.append(current_answer.strip())
                                    # Start a new answer
                                    current_answer = answer_line.strip()
                                    current_letter = answer_line[0]
                                else:
                                    # Continue the current answer
                                    current_answer += " " + answer_line.strip()
                            
                            # Don't forget the last answer
                            if current_answer:
                                answer_list.append(current_answer.strip())
                        else:
                            # Try comma separation or brackets as fallback
                            import re
                            answer_pattern = re.compile(r'([A-D]\..+?)(?=[A-D]\.|$)', re.DOTALL)
                            matches = answer_pattern.findall(choices_text + "\n")
                            if matches:
                                answer_list = [match.strip() for match in matches]
                            else:
                                # Last resort fallback to old method
                                answer_list = [choice.strip() for choice in choices_text.split(',')]
                        
                        question_info["answers"] = answer_list
                    elif in_passage:
                        passage_lines.append(line.strip())
                    elif in_question:
                        question_lines.append(line.strip())
                
                # Join the lines for passage and question
                if passage_lines:
                    passage_text = "\n".join(passage_lines)  # Preserve paragraph structure
                    question_info["passage"] = passage_text
                
                if question_lines:
                    question_text = " ".join(question_lines)
                    question_info["question"] = question_text
            
            # Additional cleaning if needed
            question_info["question"] = ' '.join(question_info["question"].split())
            
            # Clean up the answer choices to remove brackets, bullet points, etc.
            cleaned_answers = []
            for answer in question_info["answers"]:
                # Remove any trailing square brackets
                answer = re.sub(r'\]\s*$', '', answer)
                # Remove any leading square brackets
                answer = re.sub(r'^\s*\[', '', answer)
                # Remove bullet points
                answer = re.sub(r'^\s*•\s*', '', answer)
                # Clean up whitespace
                answer = answer.strip()
                cleaned_answers.append(answer)
            
            question_info["answers"] = cleaned_answers
            
            return question_info

        except Exception as e:
            print(f"Error extracting question info: {str(e)}")
            return {"passage": "", "question": "", "answers": []}

    def generate_new_question(self, original_passage, original_question, original_answers):
        try:
            # Use a more capable model for generating new questions
            model = self.get_model(task_type="default")
            
            # Create a formatted string of the original answers
            formatted_answers = ', '.join(original_answers)
            
            # Determine if we have a passage and adjust the prompt accordingly
            has_passage = original_passage and len(original_passage.strip()) > 0
            
            # Craft prompt for generating a new question of the same type but with different context
            if has_passage:
                prompt = f"""I'll provide you with an example reading passage, question, and answer choices. Please create a NEW passage and question of the SAME TYPE but with COMPLETELY DIFFERENT CONTEXT.

ORIGINAL PASSAGE: 
{original_passage}

ORIGINAL QUESTION: {original_question}

ORIGINAL CHOICES: {formatted_answers}

Requirements for the new passage and question:
1. Create a NEW PASSAGE that follows the SAME STYLE, LENGTH, and COMPLEXITY as the original
2. If the original passage is about a specific topic (e.g., science, history, literature), create a new passage about a DIFFERENT but RELATED topic
3. Maintain the same READING LEVEL and TONE as the original passage
4. Follow the EXACT SAME PATTERN and FORMAT for the question as the original
5. If the original uses specific names, places, or scenarios, use DIFFERENT ones
6. Maintain the same DIFFICULTY LEVEL
7. Create PLAUSIBLE but DIFFERENT answer choices in the same format
8. The new question should relate to the new passage in the same way the original question related to the original passage

Format your response as follows:
PASSAGE: [your new passage text]
QUESTION: [your new question]
CHOICES: [A. first choice], [B. second choice], [C. third choice], [D. fourth choice]

IMPORTANT: Just provide the new passage, question and answer choices in this format, with no additional explanation."""
            else:
                prompt = f"""I'll provide you with an example question and its answer choices. Please create a NEW question of the SAME TYPE but with COMPLETELY DIFFERENT CONTEXT.

ORIGINAL QUESTION: {original_question}
ORIGINAL CHOICES: {formatted_answers}

Requirements for the new question:
1. Follow the EXACT SAME PATTERN and FORMAT as the original question
2. Use the SAME TYPE of math or scientific concept
3. If the original uses specific numbers, use DIFFERENT numbers in your version
4. If the original uses specific names, places, or scenarios, use DIFFERENT ones
5. Maintain the same DIFFICULTY LEVEL
6. Create PLAUSIBLE but DIFFERENT answer choices in the same format
7. If the original contained a graph or diagram description, replace with a different but similar graph/diagram description

Format your response as follows:
QUESTION: [your new question]
CHOICES: [A. first choice], [B. second choice], [C. third choice], [D. fourth choice]

IMPORTANT: Just provide the new question and answer choices in this format, with no additional explanation."""

            # Generate response
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse the response to extract the new passage, question and answers
            new_question_info = {"passage": "", "question": "", "answers": []}
            
            # Parse based on whether we expect a passage or not
            if has_passage and "PASSAGE:" in response_text and "QUESTION:" in response_text and "CHOICES:" in response_text:
                # Split by the major sections
                passage_parts = response_text.split("QUESTION:")
                passage_text = passage_parts[0].replace("PASSAGE:", "").strip()
                
                remaining_text = passage_parts[1]
                question_parts = remaining_text.split("CHOICES:")
                question_text = question_parts[0].strip()
                choices_text = question_parts[1].strip() if len(question_parts) > 1 else ""
                
                new_question_info["passage"] = passage_text
                new_question_info["question"] = question_text
                new_question_info["answers"] = [choice.strip() for choice in choices_text.split(',')]
            elif "QUESTION:" in response_text and "CHOICES:" in response_text:
                # No passage format, just question and choices
                parts = response_text.split("CHOICES:")
                question_part = parts[0].strip()
                choices_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Extract the question text
                question_text = question_part.replace("QUESTION:", "").strip()
                new_question_info["question"] = question_text
                
                # Extract the choices
                new_question_info["answers"] = [choice.strip() for choice in choices_part.split(',')]
            else:
                # Fallback to line-by-line parsing
                lines = response_text.split('\n')
                passage_lines = []
                question_lines = []
                in_passage = False
                in_question = False
                
                for line in lines:
                    if line.startswith("PASSAGE:"):
                        in_passage = True
                        in_question = False
                        passage_line = line.replace("PASSAGE:", "").strip()
                        if passage_line:
                            passage_lines.append(passage_line)
                    elif line.startswith("QUESTION:"):
                        in_passage = False
                        in_question = True
                        question_lines.append(line.replace("QUESTION:", "").strip())
                    elif line.startswith("CHOICES:"):
                        in_passage = False
                        in_question = False
                        choices_text = line.replace("CHOICES:", "").strip()
                        new_question_info["answers"] = [choice.strip() for choice in choices_text.split(',')]
                    elif in_passage:
                        passage_lines.append(line.strip())
                    elif in_question:
                        question_lines.append(line.strip())
                
                # Join the lines for passage and question
                if passage_lines:
                    passage_text = "\n".join(passage_lines)  # Preserve paragraph structure
                    new_question_info["passage"] = passage_text
                
                if question_lines:
                    question_text = " ".join(question_lines)
                    new_question_info["question"] = question_text
            
            # Additional cleaning if needed
            new_question_info["question"] = ' '.join(new_question_info["question"].split())
            
            return new_question_info
            
        except Exception as e:
            print(f"Error generating new question: {str(e)}")
            return {"passage": "", "question": "", "answers": []}

    def process_uploaded_image(self, image_data):
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))

            model = self.get_model(task_type="image")
            prompt = """**Objective:** Analyze the provided image containing a graph and associated text from a standardized test question. Extract TWO distinct sets of information for different purposes:
    1.  **Graph Data:** Detailed parameters necessary to recreate the visual graph programmatically (e.g., using Matplotlib), including all integrated text elements like titles, axis labels, and legends.
    2.  **Question/Context Text:** The verbatim text content located *below* the graph area (and its legend, if applicable), comprising the passage/setup, the question itself, and the answer choices.

**Output Format:** Provide the analysis in TWO clearly separated sections below. Ensure the information in Section 1 is focused *only* on the visual elements and directly associated text labels needed for plotting. Section 2 should capture *only* the distinct text block appearing below the graph. Use markdown formatting for clarity. Be precise; if exact visual values aren't readable, estimate and note it (e.g., `~5.5` or `value (estimated)`).

---

**SECTION 1: Graph Recreation Parameters (Visual Elements & Integrated Text)**

1.  **Overall Plot Type:**
    *   Identify the primary graph type (e.g., `line chart`, `scatter plot`, `bar chart`, `stacked bar chart`, `pie chart`, `histogram`, `table`, `geometric diagram`).

2.  **Data Series & Values:**
    *   For **Charts (Line, Scatter, Bar, etc.)**:
        *   Identify each distinct data series.
        *   For each series: `Series Name/Label:` (Match legend), `Data Points:` (e.g., `[(x1, y1), ...]`, `Categories: [...], Values: [...]`), `Estimation Notes:`.
    *   For **Tables**: `Headers:`, `Rows:`.
    *   For **Geometric Shapes**: Describe shape and parameters (e.g., `center`, `radius`, `vertices`).

3.  **Axes Configuration:**
    *   **X-Axis:** `Label:` (Text on axis), `Range: [min, max]`, `Scale: linear/log`, `Ticks: [tick1, ...]`, `Tick Labels:` (If different).
    *   **Y-Axis:** `Label:`, `Range: [min, max]`, `Scale: linear/log`, `Ticks: [tick1, ...]`, `Tick Labels:`.
    *   **Secondary Y-Axis:** (If present) Describe similarly.

4.  **Visual Styling:**
    *   **For each Data Series:** `Color:`, `Line Style:`, `Marker Style:`, `Bar Style:`.
    *   **General Plot Styles:** `Grid: present/absent` (specify details), `Background Color:`.

5.  **Integrated Text Elements:** (Text physically part of or immediately adjacent to the graph visualization)
    *   `Title:` Text content (usually above).
    *   `Legend:`
        *   `Present: yes/no`.
        *   `Position:` Describe location relative to plot (e.g., `upper right`, `below plot area but above question text`).
        *   `Entries:` List the text labels (should match `Series Name/Label` from point 2).
    *   `Annotations:` Text annotations *within* the plot area or pointing directly to plot features. Provide: `Text:`, `Location: (approx. x, y)`, `Arrow: yes/no`.
    *   *(Note: Axis labels are captured in point 3)*.

---

**SECTION 2: Question/Context Text (Located Below Graph Area)**

1.  **Passage/Question Text:**
    *   Transcribe the entire block of text appearing distinctly *below* the graph area (and its legend, if the legend is positioned there). This includes any introductory passage/scenario, the specific question asked, and any source/caption integrated within this block. Ensure verbatim transcription.
2.  **Answer Choices:**
    *   Format each answer choice on its own line as follows:
        A. [complete text of first choice without bullet points]
        B. [complete text of second choice without bullet points]
    *   For each answer choice, present it as ONE COMPLETE STATEMENT
    *   Do NOT include bullet points (•) at the beginning of answer choices
    *   Do NOT include square brackets at the beginning or end of answer choices
    *   Do NOT split a single answer choice into multiple parts

---

**Important Considerations for the Model:**
*   **Section 1 vs. Section 2 Distinction:** Section 1 describes the *graph visualization* and its immediate labels/legend. Section 2 transcribes the separate *text block below* the graph containing the problem setup, question, and answers.
*   **Accuracy:** Prioritize accuracy for data points, coordinates, and verbatim text transcription. Label estimations clearly in Section 1.
*   **Structure:** Use the requested structured formats (key-value pairs, lists, coordinates) consistently.
*   **Completeness:** Capture all visible elements pertinent to each section. If no graph or no question text is present, state that clearly."""

            # Convert PIL Image to the format expected by Gemini
            response = model.generate_content([{"mime_type": "image/png", "data": image_data}, prompt])
            full_response = response.text

            # Split the response into graph analysis and question info
            sections = full_response.split("**SECTION 2: Question/Context Text")
            graph_analysis = sections[0].replace("**SECTION 1: Graph Recreation Parameters (Visual Elements & Integrated Text)**", "").strip()
            question_section = sections[1].strip() if len(sections) > 1 else ""

            # Extract question and answers
            question_info = {"question": "", "answers": [], "passage": ""}
            if question_section:
                # Split into passage/question part and answer choices part
                parts = question_section.split("**Answer Choices:**")
                if len(parts) > 1:
                    # Process the passage/question text
                    question_text_part = parts[0]
                    
                    # Look for the passage/question text section
                    if "**Passage/Question Text:**" in question_text_part:
                        passage_question_text = question_text_part.split("**Passage/Question Text:**")[1].strip()
                        
                        # Try to separate passage from question if both exist
                        # This is a heuristic - we assume questions often start with keywords like "What", "Which", "How", etc.
                        question_keywords = ["What", "Which", "How", "Why", "When", "Where", "Who", "In which", "Calculate"]
                        
                        found_question = False
                        lines = passage_question_text.split('\n')
                        
                        for i, line in enumerate(lines):
                            # Check if line starts with a question word or contains a question mark
                            if any(line.strip().startswith(keyword) for keyword in question_keywords) or '?' in line:
                                # Found the likely question
                                question_info["passage"] = '\n'.join(lines[:i]).strip()
                                question_info["question"] = '\n'.join(lines[i:]).strip()
                                found_question = True
                                break
                        
                        # If we couldn't separate, just use the whole text as the question
                        if not found_question:
                            question_info["question"] = passage_question_text
                    
                    # Process the answer choices
                    answers_part = parts[1].strip()
                    answer_lines = [line.strip() for line in answers_part.split('\n') if line.strip()]
                    
                    # Extract answers - they're typically in format like "A. Answer text"
                    answers = []
                    for line in answer_lines:
                        if line and len(line) > 2 and line[0].isalpha() and line[1] == '.':
                            answers.append(line)
                        elif line and len(line) > 2 and line[0].isdigit() and line[1] == '.':
                            answers.append(line)
                    
                    question_info["answers"] = answers

            # Generate matplotlib code without question information
            code_model = self.get_model(task_type="code")
            code_prompt = f'''Generate precise matplotlib code that will EXACTLY match the graph description with perfect accuracy:
"{graph_analysis}"

Requirements:
1. First perform detailed analysis:
   - Extract ONLY graph-related information from description
   - EXCLUDE all question text and answer choices
   - Focus ONLY on visual elements and labels that are part of the graph
   - Identify PRECISE data points and relationships
   - Record EXACT dimensions, scales, and proportions

2. Content filtering:
   - Include ONLY:
     * Graph elements (lines, bars, shapes)
     * Axis labels and scales
     * Data point labels
     * Graph-specific annotations
     * Legend if required
   - EXCLUDE:
     * Question text
     * Answer choices
     * Multiple choice options
     * Any text not directly part of the graph
     * Explanatory text outside the graph

3. Use appropriate visualization with EXACT specifications:
   For data comparison charts:
   - Bar charts:
     * Use EXACT heights for each bar
     * Maintain PRECISE bar widths and spacing
     * Position labels at EXACT coordinates
     * Example: plt.bar(x, heights, width=0.8, align='center')

   - Line/Scatter plots:
     * Plot EXACT data point coordinates
     * Ensure intersections occur at PRECISE points
     * Use exact numerical values:
        x1, y1 = [0, 20, 40], [0, 30, 60]  # First line
        x2, y2 = [0, 20, 40], [60, 30, 0]  # Second line
     * High-resolution plotting: plt.plot(x, y, '-', linewidth=1.5)

   For geometric shapes:
   - CRITICAL: Preserve EXACT vertex-label associations
     * If angle B is right angle, it MUST be at vertex B
     * If point P is at origin, label P MUST be at (0,0)
     * Match labels to vertices based on geometric properties

   - Use mathematical properties to determine label placement:
     * Right angles: Place label at the right angle vertex
     * Equal sides: Labels must match equal side endpoints
     * Special points: Origin, centroid, focus points must match labels

   - Implement precise label positioning:
     plt.text(x, y, 'B', 
             horizontalalignment='right' if x < 0 else 'left',
             verticalalignment='top' if y < 0 else 'bottom',
             transform=ax.transData,
             offset_points=(5, 5))  # Fine-tune offset

   - For specific geometric shapes:
     * Triangles: 
       - Right triangles: Right angle label MUST match description
       - Isosceles: Equal side vertex labels must match
       - Equilateral: Maintain 60° angle label positions
     * Circles:
       - Center label placement must match description
       - Radius labels at exact endpoints
     * Polygons:
       - Regular: Maintain symmetric label positions
       - Irregular: Match label positions to vertex properties

   - Verify geometric properties:
     * Check angle types match labels (right, acute, obtuse)
     * Verify parallel/perpendicular sides have correct labels
     * Confirm special points (centroids, foci) match labels
     * Ensure equal parts have corresponding labels

   For statistical visualizations:
   - Histograms: Use exact bin edges and counts
   - Box plots: Show precise quartile positions
   - Pie charts: Use exact percentage values
   - Error bars: Show exact uncertainty ranges

4. Implement precise positioning:
   - Labels and annotations:
     plt.annotate('Label', xy=(x, y), xytext=(0, 0),
                 textcoords='offset points',
                 ha='center', va='center')
   - Axis control:
     * Set exact limits: plt.xlim(xmin, xmax)
     * Use precise ticks: plt.xticks([exact_positions])
     * Maintain proper aspect ratio when needed
   - Legend placement:
     * Use exact positioning: plt.legend(loc='upper left', bbox_to_anchor=(x, y))

5. Ensure data accuracy:
   - Maintain exact numerical precision
   - Calculate precise:
     * Values and measurements
     * Proportions and ratios
     * Intervals and gaps
     * Statistical measures
   - Use exact scales and ranges
   - Preserve data relationships

6. Quality control checks:
   - Verify ONLY graph elements are included
   - Confirm NO question text or answers are present
   - Validate ALL labels and annotations are graph-related
   - Check ALL relationships and proportions
   - Ensure proper scaling across ALL dimensions
   - VERIFY label-vertex associations match description

Generate matplotlib code that reproduces ONLY the graph with perfect accuracy.
Focus on EXACT numerical values and precise positioning of graph elements ONLY.'''

            code_response = code_model.generate_content(code_prompt)
            self.current_description = f"Graph Analysis:\n{graph_analysis}\n\nSuggested Matplotlib Code:\n{code_response.text}"

            return {
                "description": self.current_description,
                "question_info": question_info
            }

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

graph_generator = WebGraphGenerator()

@app.route('/api/generate', methods=['POST'])
def generate_graph():
    try:
        if not graph_generator.setup_api():
            return jsonify({'error': 'API key not configured. Please add your Google API key to config.json'}), 401

        data = request.json
        description = data.get('description', '')
        if not description:
            return jsonify({'error': 'No description provided'}), 400

        buf = graph_generator.generate_graph_from_description(description)
        if buf is None:
            return jsonify({'error': 'Failed to generate graph'}), 500

        # Convert plot to base64
        img_str = base64.b64encode(buf.getvalue()).decode()
        return jsonify({'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        image = Image.open(image_file)

        # Convert to format suitable for Gemini's API
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        image_data = buf.getvalue()

        # Process image to extract graph information
        result = graph_generator.process_uploaded_image(image_data)
        
        if result is None:
            result = {"description": "", "question_info": {"question": "", "answers": []}}
        
        # In parallel, extract question and answer choices using the dedicated method
        question_info = graph_generator.extract_question_info(image_data)
        
        # Update the result with the question info from the dedicated model
        if result:
            # Store the original description for graph generation
            graph_description = result.get('description', '')
            
            # Replace the question info with our dedicated extraction
            result['question_info'] = question_info
            
            # Generate graph from the description
            if graph_description:
                graph_buf = graph_generator.generate_graph_from_description(graph_description)
                if graph_buf:
                    # Convert plot to base64
                    graph_img_str = base64.b64encode(graph_buf.getvalue()).decode()
                    result['graph'] = graph_img_str

        return jsonify(result)

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/correct', methods=['POST'])
def correct_graph():
    try:
        data = request.json
        correction = data.get('correction', '')
        current_description = data.get('current_description', '')

        if not correction or not current_description:
            return jsonify({'error': 'Missing correction or current description'}), 400

        # Create enhanced prompt that maintains the original graph while applying corrections
        enhanced_prompt = f'''Modify this graph description to incorporate the following correction, while maintaining all other aspects:

Original description:
{current_description}

Requested correction:
{correction}

Generate a revised description that precisely incorporates this correction while keeping all other elements the same.'''

        # Generate new description using the appropriate model
        model = graph_generator.get_model(task_type="default")
        response = model.generate_content(enhanced_prompt)
        new_description = response.text.strip()

        # Generate new graph with corrected description
        buf = graph_generator.generate_graph_from_description(new_description)
        if buf is None:
            return jsonify({'error': 'Failed to generate corrected graph'}), 500

        # Convert plot to base64
        img_str = base64.b64encode(buf.getvalue()).decode()
        return jsonify({
            'image': img_str,
            'description': new_description
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle-model-preference', methods=['POST'])
def toggle_model_preference():
    try:
        data = request.json
        preference = data.get('preference')
        
        if preference not in ['slow', 'fast']:
            return jsonify({'error': 'Invalid preference. Must be "slow" or "fast"'}), 400
        
        # Update the model preference
        graph_generator.model_preference = preference
        success = graph_generator.save_model_preference()
        
        if not success:
            return jsonify({'error': 'Failed to save model preference'}), 500
        
        return jsonify({
            'status': 'success',
            'message': f'Model preference set to {preference}',
            'preference': preference
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-model-preference', methods=['GET'])
def get_model_preference():
    return jsonify({
        'preference': graph_generator.model_preference
    })

@app.route('/api/get-api-key', methods=['GET'])
def get_api_key():
    """Returns information about whether an API key is set (not the actual key)"""
    has_api_key = graph_generator.api_key is not None and graph_generator.api_key.strip() != ""
    return jsonify({
        'hasApiKey': has_api_key
    })

@app.route('/api/update-api-key', methods=['POST'])
def update_api_key():
    """Updates the API key with a new value"""
    try:
        data = request.json
        new_api_key = data.get('api_key', '').strip()
        
        if not new_api_key:
            return jsonify({
                'status': 'error',
                'error': 'API anahtarı boş olamaz'
            }), 400
            
        success, error_message = graph_generator.update_api_key(new_api_key)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'API anahtarı başarıyla güncellendi'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': error_message
            }), 400
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/generate-new-question', methods=['POST'])
def generate_new_question():
    try:
        # Check if the API key is configured
        if not graph_generator.api_key:
            return jsonify({'error': 'API key not configured. Please add your Google API key to config.json'}), 401

        # Validate that an image was provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        
        # Validate the file is not empty
        if image_file.filename == '':
            return jsonify({'error': 'Empty file provided'}), 400
            
        try:
            # Try to open the image to confirm it's valid
            image = Image.open(image_file)
            
            # Reset the file pointer after reading
            image_file.seek(0)
            
            # Convert to format suitable for Gemini's API
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            image_data = buf.getvalue()
        except Exception as img_err:
            print(f"Error processing image: {str(img_err)}")
            return jsonify({'error': f'Invalid image file: {str(img_err)}'}), 400

        # First extract the original question and answers from the image
        try:
            question_info = graph_generator.extract_question_info(image_data)
            
            if not question_info['question']:
                return jsonify({'error': 'Could not extract question from the image'}), 400
        except Exception as extract_err:
            print(f"Error extracting question: {str(extract_err)}")
            return jsonify({'error': f'Failed to extract question: {str(extract_err)}'}), 500
            
        # Generate a new question based on the extracted question and passage
        try:
            new_question_info = graph_generator.generate_new_question(
                question_info.get('passage', ''), 
                question_info['question'], 
                question_info['answers']
            )
            
            if not new_question_info['question']:
                return jsonify({'error': 'Failed to generate new question'}), 500
        except Exception as gen_err:
            print(f"Error generating new question: {str(gen_err)}")
            return jsonify({'error': f'Failed to generate new question: {str(gen_err)}'}), 500
        
        # Include both original and new content in the response
        result = {
            'original_passage': question_info.get('passage', ''),
            'original_question': question_info['question'],
            'original_answers': question_info['answers'],
            'new_passage': new_question_info.get('passage', ''),
            'new_question': new_question_info['question'],
            'new_answers': new_question_info['answers']
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error in generate_new_question: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))