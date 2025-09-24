# lightning_module.py (версия для бинарной классификации с f1_macro)
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
    BinaryAUROC
)


class MedicalBinaryClassificationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        # Метрики для бинарной классификации
        self.val_accuracy = BinaryAccuracy()
        self.val_f1 = BinaryF1Score(average='macro')  # f1_macro для бинарной классификации
        self.val_recall = BinaryRecall(average='macro')
        self.val_precision = BinaryPrecision(average='macro')
        self.val_auroc = BinaryAUROC()

        # Метрики для обучения
        self.train_accuracy = BinaryAccuracy()
        self.train_f1 = BinaryF1Score(average='macro')

        # Для накопления val_loss
        self.validation_step_outputs = []

        # Сохраняем гиперпараметры
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        volumes, labels = batch
        outputs = self(volumes)
        loss = self.criterion(outputs, labels)

        # Вычисляем и логируем метрики на тренировке
        train_preds = torch.argmax(outputs, dim=1)
        self.train_accuracy.update(train_preds, labels)
        self.train_f1.update(train_preds, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # Логируем метрики обучения в конце эпохи
        train_acc = self.train_accuracy.compute()
        train_f1 = self.train_f1.compute()

        self.log('train_acc', train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', train_f1, on_epoch=True, prog_bar=True, logger=True)

        # Сбрасываем метрики обучения
        self.train_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        volumes, labels = batch
        outputs = self(volumes)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        # Обновляем все метрики
        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_auroc.update(torch.softmax(outputs, dim=1)[:, 1], labels)

        # Сохраняем loss для усреднения в конце эпохи
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        # Вычисляем и логируем все агрегированные метрики
        val_acc = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()
        val_recall = self.val_recall.compute()
        val_precision = self.val_precision.compute()
        val_auroc = self.val_auroc.compute()

        # Усредняем val_loss
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()

        # Логируем все метрики в конце эпохи валидации
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', val_f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', val_recall, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', val_precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auroc', val_auroc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', avg_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Выводим метрики в консоль
        print(f"\n=== Validation Metrics (Epoch {self.current_epoch}) ===")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1 Macro: {val_f1:.4f}")
        print(f"Val Recall Macro: {val_recall:.4f}")
        print(f"Val Precision Macro: {val_precision:.4f}")
        print(f"Val AUROC: {val_auroc:.4f}")
        print("=" * 50)

        # Сбрасываем внутреннее состояние метрик и очищаем накопленные outputs
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_auroc.reset()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Максимизируем f1_macro
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # максимизируем f1_macro
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1',  # мониторим val_f1_macro для изменения lr
                'interval': 'epoch'
            }
        }