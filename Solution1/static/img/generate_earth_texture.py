from PIL import Image, ImageDraw
import numpy as np

def create_earth_texture(size=1024):
    # Create a new image with a dark blue background
    img = Image.new('RGB', (size, size), (10, 20, 40))
    draw = ImageDraw.Draw(img)
    
    # Draw land masses (simplified)
    width, height = size, size
    center = (width // 2, height // 2)
    radius = min(center) * 0.9
    
    # Draw Earth with a blue gradient
    for y in range(height):
        for x in range(width):
            # Calculate distance from center
            dx = x - center[0]
            dy = y - center[1]
            dist = (dx*dx + dy*dy) ** 0.5
            
            if dist <= radius:
                # Base blue color
                r, g, b = 20, 60, 120
                
                # Add some green for land
                if (x + y) % 100 < 50 and dist > radius * 0.2:
                    g += 50
                    r += 10
                
                # Add some white for clouds
                if (x * y) % 300 < 5:
                    r = g = b = 200
                
                # Darken edges for 3D effect
                edge_factor = 1 - (dist / radius) * 0.5
                r = int(r * edge_factor)
                g = int(g * edge_factor)
                b = int(b * edge_factor)
                
                img.putpixel((x, y), (r, g, b))
    
    return img

if __name__ == "__main__":
    earth = create_earth_texture()
    earth.save("d:\\Solution1\\Solution1\\static\\img\\earth-texture.jpg", quality=95)
    print("Earth texture generated successfully!")
