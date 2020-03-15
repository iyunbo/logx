from src.features import parse, preprocess
from src.models import train_model, predict_model


def main():
    csv_file = parse.run()

    dataloader, num_classes = preprocess.run(csv_file)

    model_file = train_model.run(dataloader, num_classes)

    predict_model.run(num_classes, model_file)


if __name__ == "__main__":
    main()
