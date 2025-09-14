"""
Data preparation script for Kanji-English dataset
Downloads KANJIDIC2 XML and KanjiVG SVG files, processes them into image-text pairs
"""

import os
import gzip
import requests
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import io
import subprocess
import tempfile
import numpy as np
import re

class KanjiDatasetProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.kanjidic_url = "https://www.edrdg.org/kanjidic/kanjidic2.xml.gz"
        self.kanjivg_url = "https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz"
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        
    def download_file(self, url, filename):
        """Download file with progress bar"""
        print(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        filepath = os.path.join(self.data_dir, "raw", filename)
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                bar.update(size)
        
        return filepath
    
    def extract_gzip(self, gz_path):
        """Extract gzipped file"""
        output_path = gz_path[:-3]  # Remove .gz extension
        print(f"Extracting {gz_path}...")
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        return output_path
    
    def parse_kanjidic2(self, xml_path):
        """Parse KANJIDIC2 XML to extract kanji and English meanings"""
        print("Parsing KANJIDIC2...")
        
        kanji_data = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for character in tqdm(root.findall('character'), desc="Processing characters"):
                literal = character.find('literal')
                if literal is None:
                    continue
                
                kanji_char = literal.text
                
                # Extract English meanings
                meanings = []
                reading_meanings = character.findall('.//reading_meaning')
                for rm in reading_meanings:
                    rmgroup = rm.find('rmgroup')
                    if rmgroup is not None:
                        for meaning in rmgroup.findall('meaning'):
                            # Only get English meanings (no m_lang attribute or m_lang="en")
                            if meaning.get('m_lang') is None or meaning.get('m_lang') == 'en':
                                if meaning.text:
                                    meanings.append(meaning.text.strip().lower())
                
                if meanings:
                    kanji_data[kanji_char] = meanings
        
        except ParseError as e:
            print(f"Error parsing XML: {e}")
            return {}
        
        print(f"Extracted {len(kanji_data)} kanji characters with meanings")
        return kanji_data
    
    def parse_kanjivg_svgs(self, xml_path):
        """Parse KanjiVG XML to extract SVG data for each kanji"""
        print("Parsing KanjiVG SVGs...")
        
        svg_data = {}
        
        try:
            # Read the file and look for the structure
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse with ElementTree
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Debug: print root element info
            print(f"Root tag: {root.tag}")
            print(f"Root attributes: {root.attrib}")
            
            # Find all kanji elements - try different approaches
            kanji_elements = []
            
            # Try different XPath expressions
            xpath_patterns = [
                './/kanji',
                './/*[@id]',
                './/g[@id]'
            ]
            
            for pattern in xpath_patterns:
                elements = root.findall(pattern)
                if elements:
                    print(f"Found {len(elements)} elements with pattern: {pattern}")
                    kanji_elements = elements
                    break
            
            if not kanji_elements:
                # Try without namespace
                for elem in root.iter():
                    if 'kanji' in elem.tag.lower() or (elem.get('id') and elem.get('id').startswith('kvg:kanji')):
                        kanji_elements.append(elem)
            
            print(f"Processing {len(kanji_elements)} kanji elements...")
            
            for kanji_elem in tqdm(kanji_elements, desc="Processing SVGs"):
                kanji_id = kanji_elem.get('id', '')
                
                # Extract the kanji character from various ID formats
                kanji_char = None
                
                if 'kvg:kanji_' in kanji_id:
                    unicode_hex = kanji_id.split('kvg:kanji_')[1].split('-')[0]
                    try:
                        kanji_char = chr(int(unicode_hex, 16))
                    except (ValueError, IndexError):
                        continue
                elif kanji_id.startswith('kanji_'):
                    unicode_hex = kanji_id[6:].split('-')[0]
                    try:
                        kanji_char = chr(int(unicode_hex, 16))
                    except (ValueError, IndexError):
                        continue
                
                if not kanji_char:
                    continue
                
                # Extract all path elements (strokes)
                paths = []
                
                # Look for path elements in all descendants
                for path_elem in kanji_elem.iter():
                    if path_elem.tag.endswith('path') or 'path' in path_elem.tag:
                        path_d = path_elem.get('d')
                        if path_d:
                            paths.append(path_d)
                
                if paths:
                    svg_data[kanji_char] = paths
        
        except (ParseError, Exception) as e:
            print(f"Error parsing KanjiVG XML: {e}")
            print("Attempting alternative parsing method...")
            
            # Alternative parsing using regex
            try:
                with open(xml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find kanji elements using regex
                kanji_pattern = r'<(?:\w+:)?kanji[^>]+id="[^"]*kanji[_:]([0-9a-fA-F]+)[^"]*"[^>]*>(.*?)</(?:\w+:)?kanji>'
                kanji_matches = re.findall(kanji_pattern, content, re.DOTALL)
                
                for unicode_hex, kanji_content in tqdm(kanji_matches, desc="Processing SVGs (regex)"):
                    try:
                        kanji_char = chr(int(unicode_hex, 16))
                        
                        # Extract path elements
                        path_pattern = r'<(?:\w+:)?path[^>]+d="([^"]+)"[^>]*/?>'
                        paths = re.findall(path_pattern, kanji_content)
                        
                        if paths:
                            svg_data[kanji_char] = paths
                    except ValueError:
                        continue
            
            except Exception as e2:
                print(f"Alternative parsing also failed: {e2}")
                return {}
        
        print(f"Extracted SVG data for {len(svg_data)} kanji characters")
        return svg_data
    
    def create_svg_string(self, paths, size=256):
        """Create complete SVG string from path data"""
        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 109 109">
<g stroke="#000000" stroke-width="3" fill="none">'''
        
        path_strings = []
        for path in paths:
            path_strings.append(f'<path d="{path}"/>')
        
        svg_footer = '''</g></svg>'''
        
        return svg_header + ''.join(path_strings) + svg_footer
    
    def svg_to_image(self, svg_string, size=256):
        """Convert SVG string to PIL Image using alternative method"""
        try:
            # Create a temporary SVG file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False, encoding='utf-8') as f:
                f.write(svg_string)
                svg_path = f.name
            
            try:
                # Try using Inkscape if available (fallback method)
                png_path = svg_path.replace('.svg', '.png')
                result = subprocess.run([
                    'inkscape', '--export-type=png', f'--export-filename={png_path}',
                    f'--export-width={size}', f'--export-height={size}', svg_path
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(png_path):
                    image = Image.open(png_path).convert('RGB')
                    os.unlink(png_path)
                    os.unlink(svg_path)
                    return self._process_kanji_image(image)
                else:
                    raise Exception("Inkscape conversion failed")
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: Create a simple representation using PIL drawing
                os.unlink(svg_path)
                return self._create_simple_kanji_image(svg_string, size)
                
        except Exception as e:
            print(f"Error converting SVG to image: {e}")
            return None
    
    def _process_kanji_image(self, image):
        """Process image to ensure black strokes on white background"""
        image_array = np.array(image)
        
        # Make background white and text black
        white_bg = np.ones_like(image_array) * 255
        
        # Find non-white pixels (assume they are the kanji strokes)
        mask = (image_array.sum(axis=2) < 765)  # Not pure white
        white_bg[mask] = [0, 0, 0]  # Make strokes black
        
        return Image.fromarray(white_bg.astype(np.uint8))
    
    def _create_simple_kanji_image(self, svg_string, size=256):
        """Create a simple kanji-like image when SVG conversion fails"""
        # Create a white image
        image = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(image)
        
        # Extract path data and create simple strokes
        path_data = re.findall(r'd="([^"]+)"', svg_string)
        
        if path_data:
            # Parse simple path commands (very basic parser)
            for path in path_data:
                self._draw_simple_path(draw, path, size)
        
        return image
    
    def _draw_simple_path(self, draw, path_d, size):
        """Draw a very simple representation of SVG path"""
        try:
            # Extract move and line commands (very simplified)
            commands = re.findall(r'[ML]\s*[\d\.\s,]+', path_d, re.IGNORECASE)
            
            points = []
            current_pos = [0, 0]
            
            for cmd in commands:
                cmd_type = cmd[0].upper()
                coords = re.findall(r'[\d\.]+', cmd[1:])
                
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    # Scale from SVG coordinate system (0-109) to image size
                    x = (x / 109.0) * size
                    y = (y / 109.0) * size
                    
                    if cmd_type == 'M':  # Move to
                        if points:  # Draw previous path
                            if len(points) > 1:
                                draw.line(points, fill='black', width=3)
                        points = [(x, y)]
                        current_pos = [x, y]
                    elif cmd_type == 'L':  # Line to
                        points.append((x, y))
                        current_pos = [x, y]
            
            # Draw final path
            if len(points) > 1:
                draw.line(points, fill='black', width=3)
                
        except Exception as e:
            # If parsing fails, draw a simple placeholder
            draw.rectangle([size//4, size//4, 3*size//4, 3*size//4], outline='black', width=3)
    
    def create_dataset(self, image_size=256):
        """Create the complete dataset"""
        print("Starting dataset creation...")
        
        # Download files
        kanjidic_gz = self.download_file(self.kanjidic_url, "kanjidic2.xml.gz")
        kanjivg_gz = self.download_file(self.kanjivg_url, "kanjivg-20220427.xml.gz")
        
        # Extract files
        kanjidic_xml = self.extract_gzip(kanjidic_gz)
        kanjivg_xml = self.extract_gzip(kanjivg_gz)
        
        # Parse data
        kanji_meanings = self.parse_kanjidic2(kanjidic_xml)
        kanji_svgs = self.parse_kanjivg_svgs(kanjivg_xml)
        
        # Create dataset by matching kanji characters
        dataset = []
        images_saved = 0
        
        print("Creating image-text pairs...")
        
        for kanji_char in tqdm(kanji_meanings.keys(), desc="Processing kanji"):
            if kanji_char in kanji_svgs:
                meanings = kanji_meanings[kanji_char]
                svg_paths = kanji_svgs[kanji_char]
                
                # Create SVG string
                svg_string = self.create_svg_string(svg_paths, image_size)
                
                # Convert to image
                image = self.svg_to_image(svg_string, image_size)
                
                if image is not None:
                    # Save image
                    image_filename = f"kanji_{ord(kanji_char):05x}.png"
                    image_path = os.path.join(self.data_dir, "images", image_filename)
                    image.save(image_path, "PNG")
                    
                    # Create text description (join multiple meanings)
                    text_description = ", ".join(meanings[:3])  # Use up to 3 meanings
                    
                    dataset.append({
                        "kanji": kanji_char,
                        "image_path": image_path,
                        "text": text_description,
                        "meanings": meanings
                    })
                    
                    images_saved += 1
        
        print(f"Successfully created {len(dataset)} kanji image-text pairs")
        print(f"Saved {images_saved} images")
        
        # Save dataset metadata
        dataset_path = os.path.join(self.data_dir, "kanji_dataset.json")
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset saved to {dataset_path}")
        return dataset

if __name__ == "__main__":
    processor = KanjiDatasetProcessor()
    dataset = processor.create_dataset(image_size=256)
