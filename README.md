# Receipt Reader

## Inspiration

In a world where data is everywhere, we want get hold of our data by **automatically** extracting information from store receipts.

## What it does

It's simple. You just give a black and white perspective corrected receipt (Dropbox is excellent for this) and it will extract whatever relevant information it can from it. So far we have store name, date of purchase, total price paid and purchased items, including individual prices. We don't really care about the VAT class, but that's also there if you want it.

## How it does it 
Using classical image processing methods, we are able to **separate the image into rectangles** and then run OCR on top to detect text. We then **detect the store** by matching the store logo present on the receipt to a dictionary of store logos. The **date and total price** are found through regex searches through the entire receipt text. Based on the store, we employ different strategies to **detect the items** bought.

##Accomplishments that we're proud of
We have a fairly robust program that we developed with mostly classical computer vision methods that fulfills our needs as users. We will definitely use it!

## Challenges we ran into
OCR is not perfect, and neither are our images. We managed to remove a fair amount of noise, but there are still some images with worse artefacts which will require more aggressive denoising techniques.

Receipts vary a lot in formats, which means that a "one-size fits all" solution requires more advanced techniques.

While the similarity metric used to compare logos works well in general, it is not robust to certain types of distortion, for example rotations. There are methods out there that can cope with these problems.

## Future work
Integration with web and mobile platforms, speed and accuracy improvements.
