//
//  SUImagePickerView.swift
//  rUnet
//
//  Created by iPhoneApps on 2021-03-13.
//

import Foundation
import SwiftUI
import UIKit

// Referred the code from https://betterprogramming.pub/how-to-pick-an-image-from-camera-or-photo-library-in-swiftui-a596a0a2ece from line# 13 to 57
struct CustomUImagePickerView: UIViewControllerRepresentable {
    //
    
    var sourceType: UIImagePickerController.SourceType = .photoLibrary
    @Binding var image: Image? // Used to share common data between photo library and camera i.e between two Views
    @Binding var isPresented: Bool // to check
    
    // The custom class Coordinator is initialized and returned
    func makeCoordinator() -> ImagePickerViewCoordinator {
        
        return ImagePickerViewCoordinator(image: $image, isPresented: $isPresented)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let pickerController = UIImagePickerController()
        pickerController.sourceType = sourceType
        pickerController.delegate = context.coordinator
        return pickerController
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {

    }
}

class ImagePickerViewCoordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    
    @Binding var image: Image?
    @Binding var isPresented: Bool
    
    init(image: Binding<Image?>, isPresented: Binding<Bool>) {
        self._image = image
        self._isPresented = isPresented
    }
    
    // A predefined function of UIImagePickerControllerDelegate used to display image taken by Camera
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            self.image = Image(uiImage: image)
        }
        self.isPresented = false
    }
    
    // To exit the Camera view we use this function
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        self.isPresented = false
    }
}
