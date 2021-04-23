# rUnet-iOS-app

### ContentView.swift file
      + Run the app on a device, this file is responsible for showing an Image to the user and a button to apply a filter when tapped.
      + Also, responsible for showing options to start a Camera or to choose an image from the Photo Library to the user when tapped on the image.
      + When "Apply Filter" button is tapped, in the button action function, we make few converstions on the image by 
      using ToUIImage.swift and ImageProcessor.swift files to give input to the CoreML model as it takes CVPixelBuffer type as an input. 
      
### SUImagePickerView.swift file
      + The file is responsible in managing the image obtained from Camera or Photo Library. 
      To choose or retake the image or go back to the main screen by pressing the cancel button.
      
      
### ToUIImage.swift
      + The file is responsible for converting SwiftUI (Newest Framework) image type to UIImage type of UIKit (Old Framework)
      + Then we use this image to convert to CGImage of UIKit in ContentView.swift file


### ImageProcessor.swift
      + The file is responsible to convert the obtained CGImage in to CVPixelBuffer type to give input to the CoreML model.                                                                                
  
  
### ContentView.swift file (This is the same file described inititally)
      + Then, we define the CoreML model.
      + We then make a prediction on the model using the CVPixelBuffer obtained from ImageProcessor.swft
      
### FilterModel.mlmodel
      + It is the CoreML model using which we are trying to apply filter on the image
