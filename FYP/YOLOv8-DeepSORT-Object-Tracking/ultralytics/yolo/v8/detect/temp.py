import cv2
import firebase_admin
from firebase_admin import credentials, firestore
import json
import tkinter as tk
from tkinter import simpledialog

# List to store parking lot coordinates and names
parking_lots = []

# Variables to store the current drawing state
drawing = False
ix, iy = -1, -1

def get_parking_lot_name():
    """Get the parking lot name from the user using a dialog box."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    parking_lot_name = simpledialog.askstring("Input", "Enter parking lot name:")
    root.destroy()  # Destroy the Tkinter root window after getting input
    return parking_lot_name

def initialize_firebase():
    """Initialize the Firebase connection."""
    # Replace 'path/to/serviceAccountKey.json' with the path to your downloaded service account key
    cred = credentials.Certificate('./iparkfinder-d049e-firebase-adminsdk-oydmw-fdd288ecd1.json')
    firebase_admin.initialize_app(cred)
    return firestore.client()

def update_database(db, parking_lots):
    """Update the parking lot data in Firebase Firestore."""
    for parking_lot in parking_lots:
        name, _ = parking_lot  # We only need the name for the database
        # Update Firestore
        doc_ref = db.collection('ParkingLot').document(name)
        doc_ref.set({
            'level': 2,
            'vacant': True
        })

def load_parking_lots():
    """Load parking lot data from the JSON file."""
    global parking_lots
    try:
        with open("parking_lots.json", "r") as f:
            parking_lots = json.load(f)
            print(f"Loaded {len(parking_lots)} parking lots from 'parking_lots.json'.")
    except FileNotFoundError:
        print("No 'parking_lots.json' file found. Starting with an empty list.")

def draw_existing_parking_lots():
    """Draw the existing parking lots on the image."""
    for parking_lot in parking_lots:
        name = parking_lot[0]  # First element is the parking lot name
        x1, y1, x2, y2 = parking_lot[1:]  # The remaining elements are the coordinates
        # Draw the rectangle on the original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, parking_lots, img

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the rectangle
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # If drawing, display the rectangle being drawn
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Parking Lots", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finalize the rectangle
        drawing = False
        # Ask for the name of the parking lot
        parking_lot_name = get_parking_lot_name()

        if parking_lot_name:
            # Save the parking lot data if the name is provided
            parking_lots.append([parking_lot_name, ix, iy, x, y])
            # Draw the final rectangle on the image
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.putText(img, parking_lot_name, (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # If no name is provided, reset to original state without the current rectangle
            img[:] = img_original.copy()
            draw_existing_parking_lots()

        cv2.imshow("Draw Parking Lots", img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove a parking lot if right-click is detected within a rectangle
        for i, (name, x1, y1, x2, y2) in enumerate(parking_lots):
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"Removing parking lot '{name}' at coordinates {(x1, y1, x2, y2)}")
                parking_lots.pop(i)
                # Redraw the image without the removed rectangle
                img[:] = img_original.copy()
                draw_existing_parking_lots()
                break

import os

# Initialize Firebase
db = initialize_firebase()

image_path = "./parking_lots.png"
if not os.path.exists(image_path):
    print("Image file does not exist:", image_path)
    exit()

# Load the image without resizing
img = cv2.imread(image_path)

# Make a copy of the original image
img_original = img.copy()

# Load existing parking lot data and draw them on the original image
load_parking_lots()
draw_existing_parking_lots()

# Create a window and set the mouse callback function
cv2.namedWindow("Draw Parking Lots")
cv2.setMouseCallback("Draw Parking Lots", draw_rectangle)

print("Draw rectangles for each parking lot and name them.")
print("Right-click to remove a box. Press 's' to save and 'q' to quit without saving.")

# Display the image and wait for user actions
while True:
    cv2.imshow("Draw Parking Lots", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the parking lot data to a JSON file
        with open("parking_lots.json", "w") as f:
            json.dump(parking_lots, f)
        print(f"Saved {len(parking_lots)} parking lots to 'parking_lots.json'.")
        # Update the Firestore database
        update_database(db, parking_lots)
        print("Updated the Firebase Firestore database.")
        break

cv2.destroyAllWindows()
