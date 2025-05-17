import os
import requests
from pathlib import Path
import base64
import json

def convert_mermaid_to_png(input_file, output_file):
    """Convert a Mermaid diagram to PNG using the Mermaid Live Editor API"""
    try:
        # Read the Mermaid content
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Extract the Mermaid code from the markdown
        mermaid_code = content.strip('```mermaid\n').strip('```')
        
        # Prepare the request to Mermaid Live Editor API
        url = "https://mermaid.ink/img/"
        encoded_graph = base64.b64encode(mermaid_code.encode()).decode()
        params = {
            "type": "png",
            "theme": "default",
            "backgroundColor": "transparent",
            "scale": 1.5  # Increase scale for better quality
        }
        
        # Make the request
        response = requests.get(f"{url}{encoded_graph}", params=params)
        response.raise_for_status()
        
        # Save the PNG file
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")
        # Try alternative method for sequence diagrams
        if "sequenceDiagram" in mermaid_code:
            try:
                # Use a different API endpoint for sequence diagrams
                url = "https://mermaid.ink/svg/"
                response = requests.get(f"{url}{encoded_graph}", params=params)
                response.raise_for_status()
                
                # Save as SVG instead
                svg_output = str(output_file).replace('.png', '.svg')
                with open(svg_output, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully converted {input_file} to {svg_output}")
            except Exception as e2:
                print(f"Error in alternative conversion: {str(e2)}")

def main():
    # Create output directory if it doesn't exist
    output_dir = Path('diagrams/png')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all Mermaid diagram files
    diagram_dir = Path('diagrams')
    mermaid_files = list(diagram_dir.glob('*.md'))
    
    # Convert each file
    for mermaid_file in mermaid_files:
        output_file = output_dir / f"{mermaid_file.stem}.png"
        convert_mermaid_to_png(str(mermaid_file), str(output_file))

if __name__ == '__main__':
    main() 