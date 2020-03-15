from src.data import make_dataset
from src.models import train_model, predict_model


def main():
    dataloader, num_classes = make_dataset.main()

    model_file = train_model.main(dataloader, num_classes)

    predict_model.main(num_classes, model_file)


if __name__ == "__main__":
    main()
