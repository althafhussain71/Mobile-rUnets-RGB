//
//  ContentView.swift
//  rUnet
//
//  Created by iPhoneApps on 2021-03-13.
//

import SwiftUI
import CoreML
import Vision

// Referred the code from https://betterprogramming.pub/how-to-pick-an-image-from-camera-or-photo-library-in-swiftui-a596a0a2ece from line# 13 to 42
struct ContentView: View {
    
    @State private var image: Image? = Image("image_39")
        @State private var presentImagePicker = false
        @State private var presentActionScheet = false
        @State private var presentCamera = false
    
    var body: some View {
        
        VStack{
            Text("Tap Image to select Camera or Photo Library")
                .padding()
            image!
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .clipShape(Rectangle())
                        .overlay(Rectangle().stroke(Color.white, lineWidth: 4))
                        .shadow(radius: 10)
                        .onTapGesture {
                            self.presentActionScheet = true }
                        .sheet(isPresented: $presentImagePicker) {
                            CustomUImagePickerView(sourceType: self.presentCamera ? .camera : .photoLibrary, image: self.$image, isPresented: self.$presentImagePicker)
                    }.actionSheet(isPresented: $presentActionScheet) { () -> ActionSheet in
                        ActionSheet(title: Text("Select your mode"), message: Text("Please select your mode of image"), buttons: [ActionSheet.Button.default(Text("Camera"), action: {
                            self.presentImagePicker = true
                            self.presentCamera = true
                        }), ActionSheet.Button.default(Text("Photo Library"), action: {
                            self.presentImagePicker = true
                            self.presentCamera = false
                        }), ActionSheet.Button.cancel()])
                    }
            // Function is called when Apply Filter Button is clicked, here defining CoreML model and doing a prediction. Model expects CVPixelBuffer as input, hence doing few conversions like SwiftUI image to UIImage of UIKit then converting the obtained UIImage to CGImage then finally converting it to CVPixelBuffer format as we can't directly convert SwiftUI image to a CVPixelBuffer. Finally doing a prediction using the obtained CVPixelBuffer.
            Button(action: {
                
                let toUIImage: UIImage = image.asUIImage() // Calling asUIImage() to convert SwiftUI image to UIImage of UIKit
                let toCGImage = toUIImage.cgImage // Converting from UIImage to cgImage using predefined attribute of UIKit
                // Calling funtion pixelBuffer to do conversion from CGImage to CVPixelBuffer
                let pxlBuffer = ImageProcessor.pixelBuffer(forImage: toCGImage!)
                let model = FilterModel()
                // output variable has the model output in MultiArrayFloat32 type, could not convert this back to the image (Blurred Image)
                guard let output = try? model.prediction(input_1: pxlBuffer!) else {
                    return
                }
                // Will give the shape of the output 
                // let shape = output._38.shape
            }) {
                Text("Apply Filter")
            }
        }
        
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
