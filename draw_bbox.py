from PIL import Image, ImageDraw

def draw_bbox(image_path, bbox):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=2)
    return image

def save_image_with_bbox(image_path, bbox, output_path):
    original_image = Image.open(image_path)
    bbox_image = draw_bbox(image_path, bbox)
    
    # Create a new image with both original and bbox images side by side
    combined_image = Image.new('RGB', (original_image.width * 2, original_image.height))
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(bbox_image, (original_image.width, 0))
    
    # Save the combined image
    combined_image.save(output_path)
    print(f"Combined image saved to {output_path}")

if __name__ == "__main__":
    image_path = "./tes.png"
    bbox = (0, 10, 28, 50)  # (x1, y1, x2, y2)
    output_path = "./output_with_bbox.png"
    save_image_with_bbox(image_path, bbox, output_path)
