# lightning_module.py (улучшенная версия вашего кода)
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassF1Score, 
    MulticlassRecall, 
    MulticlassPrecision
)


class MedicalClassificationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, num_classes=2):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        # Метрики для валидации - добавляем все нужные метрики
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)
        self.val_recall = MulticlassRecall(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes)

        # Метрики для обучения (опционально, для более полной картины)
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes)

        # Для накопления val_loss
        self.validation_step_outputs = []
        
        # Сохраняем гиперпараметры для логгирования
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

        # Сохраняем loss для усреднения в конце эпохи
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        # Вычисляем и логируем все агрегированные метрики
        val_acc = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()
        val_recall = self.val_recall.compute()
        val_precision = self.val_precision.compute()

        # Усредняем val_loss
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()

        # Логируем все метрики в конце эпохи валидации
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', val_f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', val_recall, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', val_precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', avg_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Выводим метрики в консоль
        print(f"\n=== Validation Metrics (Epoch {self.current_epoch}) ===")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        print(f"Val Recall: {val_recall:.4f}")
        print(f"Val Precision: {val_precision:.4f}")
        print("=" * 50)

        # Сбрасываем внутреннее состояние метрик и очищаем накопленные outputs
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Добавляем scheduler с мониторингом val_f1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # максимизируем F1
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1',  # мониторим val_f1 для изменения lr
                'interval': 'epoch'
            }
        }