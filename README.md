# Getting Started

## Dependencies
1. Make sure you have Python and pip.

    1. Install venv.

        Line (Windows): ```sudo apt install python3.10-venv```

        Line (MacOS): ```brew install python3.10```

    3. Create your own envoirnment folder. Dependencies will go here.

        Line: ```python3 -m venv venv```

    4. Activite your envoirnment. Your directory should now start with (venv).

        Line: ```source venv/bin/activate```

    5. Using pip install all dependencies.

        Line: ```pip install -r requirements.txt```

2. Install the NABirds Dataset and place it in the raw_data folder: 
    - Link to dataset: https://dl.allaboutbirds.org/merlin---computer-vision--terms-of-use?submissionGuid=c834c526-f32a-4782-90e7-f1a6ca39bb30

## Running The Project
1. Run the data processor and train test split: 

    1. python scripts/process_raw_data.py

    2. python scripts/train_test_split.py

2. Train a model:

    - python src/train.py -kf number_of_desired_classes number_of_desired_epoches

    - Note: including -kf means to train using K-Fold Cross Validation

3. Run the frontend and backend:

    1. In one terminal, cd backend and run: python3 server.py 8080
    
    2. In another terminal, cd frontend and run: npm run dev

4. Enjoy BirdML.
