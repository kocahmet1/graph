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

    def setup_api(self):
        try:
            with open("config.json") as f:
                config = json.load(f)
                api_key = config.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("No API key found in config.json")
                genai.configure(api_key=api_key)
                return True
        except Exception as e:
            print(f"API setup error: {str(e)}")
            return False
        except FileNotFoundError:
            print("config.json file not found")
            return False
        except Exception as e:
            print(f"Error setting up API: {str(e)}")
            return False

    def generate_graph_from_description(self, description):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
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
            code = response.text

            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

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

    def process_uploaded_image(self, image_data):
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))

            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = """Analyze this graph image and provide TWO separate sections:

SECTION 1 - Graph Analysis (for recreation):
1. Graph Type: Determine if it's a table, bar chart, line chart, or geometric shape
2. Data Content:
   - For tables: Describe rows, columns, and cell values
   - For charts: List data points, axes ranges, and values
   - For shapes: Describe dimensions and coordinates
3. Visual Elements:
   - Colors used
   - Line styles (if applicable)
   - Markers or points (if applicable)
   - Grid presence
4. Labels and Text:
   - Title (if any)
   - Axis labels
   - Legend content
5. Special Features:
   - Annotations
   - Custom styling

SECTION 2 - Question Information:
1. Question Text: The complete question text
2. Answer Choices: List all answer choices

Please format your response with clear section headers."""

            # Convert PIL Image to the format expected by Gemini
            response = model.generate_content([{"mime_type": "image/png", "data": image_data}, prompt])
            full_response = response.text

            # Split the response into graph analysis and question info
            sections = full_response.split("SECTION 2 - Question Information:")
            graph_analysis = sections[0].replace("SECTION 1 - Graph Analysis (for recreation):", "").strip()
            question_section = sections[1].strip() if len(sections) > 1 else ""

            # Extract question and answers
            question_info = {"question": "", "answers": []}
            if question_section:
                lines = question_section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("1. Question Text:"):
                        question_info["question"] = line.replace("1. Question Text:", "").strip()
                    elif line.startswith("2. Answer Choices:"):
                        answers_text = line.replace("2. Answer Choices:", "").strip()
                        question_info["answers"] = [ans.strip() for ans in answers_text.split(',')]

            # Generate matplotlib code without question information
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

            code_response = model.generate_content(code_prompt)
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

        result = graph_generator.process_uploaded_image(image_data)
        return jsonify(result)

    except Exception as e:
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

        # Generate new description
        model = genai.GenerativeModel('gemini-2.0-flash')
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

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))