from PIL import Image
til = Image.new("RGB",(1000, 1000), color = 'white')

card_name = '2c'
im = Image.open('../Data/Images/Post-Processed/' + card_name + '.jpg') #25x25

# define an offset at which the image gets pasted.
x_pos = 100
y_pos = 100

til.paste(im, (x_pos ,y_pos))

til.save('../Data/Images/pics_on_canvas/' + card_name + '.png')