---- .aligncenter .bg-white
@unsplash(IWenq-4JHqo) .dark

.text-data **Synthetic Countenance**
.text-landing Everything is fake

@footer
 @h4 
 
 @div .wrap @div .span
  @button(href="https://github.com/thoppe/SyntheticCountenance") .alignleft .ghost
   ::github:: Talk repo
  @button(href="https://twitter.com/metasemantic") .ghost .alignright
   ::twitter:: @metasemantic

---- .aligncenter
@unsplash(9ITUSJZRxP8) .dark

.text-data **GANs**
## _generative adversarial networks_
<br><br><br>

## One really cool trick to generate synthetic samples.
## Used [Progressive Growing of GANs](https://github.com/tkarras/progressive_growing_of_gans)
## Celebrity dataset [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---- .aligncenter
@unsplash(9ITUSJZRxP8) .dark

# Crazy good results!
<img src="figures/good/000161.jpg">
<img src="figures/good/000165.jpg">
<img src="figures/good/000169.jpg">
<img src="figures/good/000017.jpg">
<br>
<img src="figures/good/000405.jpg">
<img src="figures/good/000415.jpg">
<img src="figures/good/000416.jpg">
<img src="figures/good/000450.jpg">
<br>
Heads up, nobody here is real!

---- .aligncenter
@unsplash(9ITUSJZRxP8) .dark
# even the failures are interesting!

<img src="figures/bad/000286.jpg">
<img src="figures/bad/000291.jpg">
<img src="figures/bad/000302.jpg">
<img src="figures/bad/000323.jpg">
<br>
<img src="figures/bad/000340.jpg">
<img src="figures/bad/000349.jpg">
<img src="figures/bad/000401.jpg">
<img src="figures/bad/000402.jpg">
<br>

---- .bg-black .align-left  .slide-top
@background_video(url="figures/sample_faces.mp4")
# Sample faces "nearby"
# use optimal transport!


---- .aligncenter
@unsplash(9ITUSJZRxP8) .dark

.text-landing math math math math math math math math
<br><br>

## Normal generator (z gives an image)
## $G(z) = I$

<br><br>
## What I wanted (get z from any image)
## $G^{-1}(I) = z$

<br><br>
## What I did (get z from a single image)
## $F_{I}^{-1}(I) \approx z$


---- .bg-black .align-left  .slide-top
@background_video(url="figures/Ali_Spittel.mp4")
#### Ali Spittel
#### Formerly GA, now at Dev.to
#### used with permission

---- .bg-black .align-left  .slide-top
@background_video(url="figures/ChristineFarr.mp4")
#### Christina Farr
#### Reporter at @CNBC.com
#### used with permission

---- .bg-black .align-left  .slide-top
@background_video(url="figures/LuisFonsi.mp4")
#### Luis Fonsi
#### That guy from _Despacito_
#### Public image

---- .bg-black .align-left  .slide-top
@background_video(url="figures/Rihanna.mp4")
#### Rihanna
#### 9-time Grammy winner
#### and Barbadian Ambassador at-Large
#### Public image

---- .aligncenter .bg-black
@unsplash(9ITUSJZRxP8) .dark
.text-data **y tho?**

## Models are a reflection of data.
## Explore the space to explore ourselves.
## Black mirror vs White mirror.

---- .aligncenter .bg-black
@unsplash(IWenq-4JHqo) 

.text-data **Thanks, you.**
.text-intro  @button(href="https://twitter.com/metasemantic") .ghost ::twitter:: @metasemantic




