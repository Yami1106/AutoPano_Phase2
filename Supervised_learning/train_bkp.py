def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    # model = HomographyModel()

    # Initialize the model and move to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {"InputSize": ImageSize, "OutputSize": 8}
    model = HomographyModel(hparams=hparams).to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Epoch 49: Val Loss: 183.1978 | with 100 epochs
    # Optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) # Epoch 49: Val Loss: 234.8409
    # Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Epoch 49: Val Loss: 211.4395 (gamma: 0.1, step-size: 10, weight-decay: 1e-4, lr: 1e-4)

    # Decay the learning rate by 10% every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=20, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.1)

    # train_path = os.path.join(BasePath, "patch_train")
    # TrainSet = HomographyDataset(train_path)
    # TrainLoader = DataLoader(
    #     TrainSet, batch_size=MiniBatchSize, shuffle=True, num_workers=8, pin_memory=True
    # )

    # --- TRAINING DATA SETUP ---
    train_path = os.path.join(BasePath, "patch_train")
    DirNamesTrain = [
        f.name.replace("_H4Pt.npy", "") for f in Path(train_path).glob("*_H4Pt.npy")
    ]
    TrainCoordinates = [
        np.load(os.path.join(train_path, f"{s}_H4Pt.npy")) for s in DirNamesTrain
    ]

    # --- VALIDATION DATA SETUP ---
    val_path = os.path.join(BasePath, "patch_val")
    DirNamesVal = [
        f.name.replace("_H4Pt.npy", "") for f in Path(val_path).glob("*_H4Pt.npy")
    ]
    ValCoordinates = [
        np.load(os.path.join(val_path, f"{s}_H4Pt.npy")) for s in DirNamesVal
    ]

    # val_path = os.path.join(BasePath, "patch_val")
    # ValSet = HomographyDataset(val_path)
    # ValLoader = DataLoader(
    #     ValSet, batch_size=MiniBatchSize, shuffle=False, num_workers=4
    # )

    # best_val_loss = float("inf")

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):

        model.train()

        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        # for PerEpochCounter, (patch_a, patch_b, CoordinatesBatch) in tqdm(range(NumIterationsPerEpoch)):
        # for PerEpochCounter, (input_tensor, CoordinatesBatch) in enumerate(
        #     tqdm(TrainLoader)
        # ):
        for PerEpochCounter in range(NumIterationsPerEpoch):
            # for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # Generate Batch from patch_train
            input_tensor, CoordinatesBatch = GenerateBatch(
                BasePath,
                DirNamesTrain,
                TrainCoordinates,
                MiniBatchSize,
                device,
                mode="train",
            )

            # Moving the patches and CoordinatesBatch to gpu
            input_tensor, CoordinatesBatch = (
                input_tensor.to(device),
                CoordinatesBatch.to(device),
            )

            Optimizer.zero_grad()
            batch = (input_tensor, CoordinatesBatch)

            PredicatedCoordinatesBatch = model.training_step(
                batch, batch_idx=PerEpochCounter
            )

            LossThisBatch = PredicatedCoordinatesBatch["loss"]
            # # Predict output with forward pass
            # PredicatedCoordinatesBatch = model(input_tensor)

            # I1Batch, CoordinatesBatch = GenerateBatch(
            #     BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
            # )

            # Predict output with forward pass
            # PredicatedCoordinatesBatch = model(I1Batch)
            # LossThisBatch = LossFn_supervised(
            #     PredicatedCoordinatesBatch, CoordinatesBatch
            # )

            LossThisBatch.backward()
            Optimizer.step()

            # FIX: Log Training Loss every iteration so you can see the curve
            Writer.add_scalar(
                "Loss/Train_Batch",
                LossThisBatch.item(),
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # result = model.validation_step(Batch)

        model.eval()
        val_results = []

        # Use a fixed number of samples for validation to save time
        NumValIters = max(1, int(len(DirNamesVal) / MiniBatchSize))
        with torch.no_grad():
            for _ in range(NumValIters):
                v_input, v_gt = GenerateBatch(
                    BasePath, DirNamesVal, ValCoordinates, MiniBatchSize, mode="val"
                )
                v_res = model.validation_step((v_input, v_gt), batch_idx=None)
                val_results.append(v_res["val_loss"])

                # Move validation batch to GPU
                input_tensor, gt = v_input.to(device), v_gt.to(device)

                # If your validation_step expects [patch_a, patch_b, gt]
                current_batch = (input_tensor, gt)

                # "val_loss": tensor
                result = model.validation_step(current_batch, batch_idx=None)
                val_results.append(result["val_loss"])

        # Calculate Average Validation Loss
        avg_val_loss = torch.stack(val_results).mean()

        # Tensorboard
        Writer.add_scalar("Loss/Validation_Epoch", avg_val_loss, Epochs)
        print(f"\nEpoch {Epochs}: Val Loss: {avg_val_loss:.4f}")

        # # Tensorboard
        # Writer.add_scalar(
        #     "LossEveryIter",
        #     result["val_loss"],
        #     Epochs * NumIterationsPerEpoch + PerEpochCounter,
        # )
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

        # Update learning rate every epoch
        scheduler.step()
        Writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], Epochs)

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CheckPointPath + "best_model.ckpt")
            print(f"New Best Model saved with Val Loss: {best_val_loss:.4f}")

        print("\n" + SaveName + " Model Saved...")
