from PIL import Image
canvas_size = (1100, 1100)
til = Image.new("RGB", canvas_size, color = 'white')
# images are of format 600x900, so in order to facilitate nice rotations and so forth we need 1081 pixels canvas
card_name = '2c'
im = Image.open('../Data/Images/Post-Processed/' + card_name + '.jpg') #25x25

# define an offset at which the image gets pasted.
midpoint = (im.size[0] / 2 , im.size[1] / 2)
x_pos = int(canvas_size[0]/2 - midpoint[0])
y_pos = int(canvas_size[1]/2 - midpoint[1])

til.paste(im, (x_pos, y_pos))

til.save('../Data/Images/pics_on_canvas/' + card_name + '.png')