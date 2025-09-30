# lightning_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional
import torchmetrics
from torchmetrics import (
    Accuracy,
    F1Score,
    AUROC,
    Precision,
    Recall,
    Specificity,
    ConfusionMatrix
)
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с несбалансированными классами.
    Особенно эффективна для медицинских данных.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MedicalClassificationModel(pl.LightningModule):
    """
    Lightning модуль для бинарной классификации медицинских изображений.
    Включает продвинутые метрики, loss функции и логирование.
    """

    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-4,
            num_classes: int = 2,
            use_weighted_loss: bool = True,
            class_weights: Optional[torch.Tensor] = None,
            use_focal_loss: bool = False,
            focal_alpha: float = 1.0,
            focal_gamma: float = 2.0,
            lr_scheduler: str = 'plateau',  # 'plateau', 'cosine', 'none'
            lr_patience: int = 5,
            lr_factor: float = 0.5,
            lr_min: float = 1e-7,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.use_weighted_loss = use_weighted_loss
        self.use_focal_loss = use_focal_loss

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Инициализация loss_fn — БЕЗ привязки к устройству!
        if self.use_focal_loss:
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            weight = self.class_weights if self.use_weighted_loss else None
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)  # weight может быть None или тензором

        # Инициализация метрик
        self._init_metrics()
        # Для хранения предсказаний и целей
        self.validation_step_outputs = []
        self.test_step_outputs = []

        #print(f"MedicalClassificationModel device: {self.device}", self.device)

    def _init_metrics(self):
        """Инициализация торчевых метрик."""

        # Общие параметры для метрик
        metric_kwargs = {
            'task': 'binary' if self.num_classes == 2 else 'multiclass',
            'num_classes': self.num_classes
        }

        # Метрики для обучения
        self.train_specificity = Specificity(**metric_kwargs)
        self.train_accuracy = Accuracy(**metric_kwargs)
        self.train_f1 = F1Score(**metric_kwargs)
        self.train_precision = Precision(**metric_kwargs)
        self.train_recall = Recall(**metric_kwargs)

        # Метрики для валидации
        self.val_accuracy = Accuracy(**metric_kwargs)
        self.val_f1 = F1Score(**metric_kwargs)
        self.val_precision = Precision(**metric_kwargs)
        self.val_recall = Recall(**metric_kwargs)
        self.val_specificity = Specificity(**metric_kwargs)
        self.val_auroc = AUROC(**metric_kwargs)

        # Метрики для тестирования
        self.test_accuracy = Accuracy(**metric_kwargs)
        self.test_f1 = F1Score(**metric_kwargs)
        self.test_precision = Precision(**metric_kwargs)
        self.test_recall = Recall(**metric_kwargs)
        self.test_specificity = Specificity(**metric_kwargs)
        self.test_auroc = AUROC(**metric_kwargs)

        # Confusion Matrix
        self.val_confusion_matrix = ConfusionMatrix(**metric_kwargs)
        self.test_confusion_matrix = ConfusionMatrix(**metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        device = next(self.model.parameters()).device
        x = x.to(device)
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, Any]:
        """Шаг обучения."""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)

        # Предсказания
        preds = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(preds, dim=1)

        # Метрики
        self.train_accuracy(pred_classes, targets)
        self.train_f1(pred_classes, targets)
        self.train_precision(pred_classes, targets)
        self.train_recall(pred_classes, targets)
        self.train_specificity(pred_classes, targets)

        # Логирование
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True)
        self.log('train_specificity', self.train_specificity, on_step=False, on_epoch=True)

        return {
            'loss': loss,
            'preds': pred_classes.detach(),
            'targets': targets.detach(),
            'probs': preds.detach()
        }

    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, Any]:
        """Шаг валидации."""
        inputs, targets = batch

        # Принудительно перемещаем на устройство модели
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)

        # Предсказания - убеждаемся что все на правильном устройстве
        probs = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        # Метрики - убеждаемся что входы на правильном устройстве
        self.val_accuracy(pred_classes.to(self.device), targets.to(self.device))
        self.val_f1(pred_classes.to(self.device), targets.to(self.device))
        self.val_precision(pred_classes.to(self.device), targets.to(self.device))
        self.val_recall(pred_classes.to(self.device), targets.to(self.device))
        self.val_specificity(pred_classes.to(self.device), targets.to(self.device))

        # AUROC требует вероятности
        if self.num_classes == 2:
            self.val_auroc(probs[:, 1].to(self.device), targets.to(self.device))
        else:
            self.val_auroc(probs.to(self.device), targets.to(self.device))

        # Confusion Matrix
        self.val_confusion_matrix(pred_classes.to(self.device), targets.to(self.device))

        # Сохраняем для детального анализа - НА CPU чтобы избежать накопления на GPU
        step_output = {
            'val_loss': loss.detach().cpu(),
            'preds': pred_classes.detach(),
            'targets': targets.detach(),
            'probs': probs.detach()
        }
        self.validation_step_outputs.append(step_output)

        # Логирование
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True, prog_bar=False)

        return step_output

    def test_step(self, batch: tuple, batch_idx: int) -> Dict[str, Any]:
        """Шаг тестирования."""
        inputs, targets = batch
        device = next(self.model.parameters()).device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)

        # Предсказания
        probs = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        # Метрики
        self.test_accuracy(pred_classes, targets)
        self.test_f1(pred_classes, targets)
        self.test_precision(pred_classes, targets)
        self.test_recall(pred_classes, targets)
        self.test_specificity(pred_classes, targets)

        # AUROC
        if self.num_classes == 2:
            self.test_auroc(probs[:, 1], targets)
        else:
            self.test_auroc(probs, targets)

        # Confusion Matrix
        self.test_confusion_matrix(pred_classes, targets)

        # Сохраняем для анализа НА ТОМ ЖЕ УСТРОЙСТВЕ
        step_output = {
            'test_loss': loss,
            'preds': pred_classes.detach(),  # Убираем .cpu()
            'targets': targets.detach(),     # Убираем .cpu()
            'probs': probs.detach()          # Убираем .cpu()
        }
        self.test_step_outputs.append(step_output)

        # Логирование
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test_specificity', self.test_specificity, on_step=False, on_epoch=True)


        return step_output

    def on_validation_epoch_end(self):
        """Вызывается в конце каждой эпохи валидации."""
        if self.validation_step_outputs:
            # ИСПРАВЛЕНИЕ: Убеждаемся, что все тензоры на одном устройстве
            device = self.device  # Используем self.device вместо next(self.model.parameters()).device
            all_preds = torch.cat([x['preds'].to(device) for x in self.validation_step_outputs])
            all_targets = torch.cat([x['targets'].to(device) for x in self.validation_step_outputs])
            all_probs = torch.cat([x['probs'].to(device) for x in self.validation_step_outputs])

            # Детализированный отчет (каждые несколько эпох)
            if self.current_epoch % 10 == 0:
                self._log_detailed_metrics(all_preds.cpu(), all_targets.cpu(), all_probs.cpu(), 'val')

            # Сохраняем метрики для использования в коллбэках
            self.validation_metrics = {
                'val_acc': self.val_accuracy.compute().item(),
                'val_f1': self.val_f1.compute().item(),
                'val_auroc': self.val_auroc.compute().item(),
                'val_precision': self.val_precision.compute().item(),
                'val_recall': self.val_recall.compute().item(),
                'val_specificity': self.val_specificity.compute().item()
            }

            # Очищаем накопленные выходы
            self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Вызывается в конце тестирования."""
        if self.test_step_outputs:
            # ИСПРАВЛЕНИЕ: Убеждаемся, что все тензоры на одном устройстве
            device = self.device  # Используем self.device вместо next(self.model.parameters()).device
            all_preds = torch.cat([x['preds'].to(device) for x in self.test_step_outputs])
            all_targets = torch.cat([x['targets'].to(device) for x in self.test_step_outputs])
            all_probs = torch.cat([x['probs'].to(device) for x in self.test_step_outputs])

            # Детализированный отчет
            self._log_detailed_metrics(all_preds.cpu(), all_targets.cpu(), all_probs.cpu(), 'test')

            # Очищаем накопленные выходы
            self.test_step_outputs.clear()

    def _log_detailed_metrics(self, preds: torch.Tensor, targets: torch.Tensor,
                              probs: torch.Tensor, stage: str):
        """Логирование детализированных метрик."""

        # Классификационный отчет
        try:
            report = classification_report(
                targets.numpy(),
                preds.numpy(),
                target_names=['Норма', 'Патология'] if self.num_classes == 2 else None,
                output_dict=True,
                zero_division=0
            )

            # Логируем классификационный отчет
            if self.logger:
                self.logger.experiment.add_text(
                    f'{stage}_classification_report',
                    str(classification_report(
                        targets.numpy(),
                        preds.numpy(),
                        target_names=['Норма', 'Патология'] if self.num_classes == 2 else None,
                        zero_division=0
                    )),
                    self.current_epoch
                )
        except Exception as e:
            self.print(f"Ошибка при создании классификационного отчета: {e}")

        # Confusion Matrix визуализация
        try:
            cm = self.val_confusion_matrix.compute() if stage == 'val' else self.test_confusion_matrix.compute()
            cm_np = cm.cpu().numpy()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_np,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Норма', 'Патология'] if self.num_classes == 2 else None,
                yticklabels=['Норма', 'Патология'] if self.num_classes == 2 else None,
                ax=ax
            )
            ax.set_xlabel('Предсказанный класс')
            ax.set_ylabel('Истинный класс')
            ax.set_title(f'Confusion Matrix ({stage.upper()})')

            if self.logger:
                self.logger.experiment.add_figure(f'{stage}_confusion_matrix', fig, self.current_epoch)

            plt.close(fig)

        except Exception as e:
            self.print(f"Ошибка при создании confusion matrix: {e}")

        # ROC кривая для бинарной классификации
        if self.num_classes == 2:
            try:
                fpr, tpr, _ = roc_curve(targets.numpy(), probs[:, 1].numpy())
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC кривая (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                        label='Случайный классификатор')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC кривая ({stage.upper()})')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)

                if self.logger:
                    self.logger.experiment.add_figure(f'{stage}_roc_curve', fig, self.current_epoch)

                plt.close(fig)

            except Exception as e:
                self.print(f"Ошибка при создании ROC кривой: {e}")

    def configure_optimizers(self):
        """Конфигурация оптимизатора и планировщика."""

        # Оптимизатор Adam с weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        # Планировщик learning rate
        if self.hparams.lr_scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',  # для F1 score
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
                verbose=True
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_f1",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }

        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=25,  # половина от максимального числа эпох
                eta_min=self.hparams.lr_min
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }

        else:
            return optimizer

    def predict_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Шаг предсказания для inference."""
        inputs, _ = batch if isinstance(batch, (tuple, list)) and len(batch) == 2 else (batch, None)

        outputs = self.forward(inputs)
        probs = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        return {
            'predictions': pred_classes,
            'probabilities': probs,
            'logits': outputs
        }

    def get_model_summary(self) -> str:
        """Возвращает краткое описание модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary = f"""
        🏥 Medical Classification Model Summary:
        • Total parameters: {total_params:,}
        • Trainable parameters: {trainable_params:,}
        • Non-trainable parameters: {total_params - trainable_params:,}
        • Model: {self.model.__class__.__name__}
        • Loss function: {'Focal Loss' if self.use_focal_loss else 'Cross Entropy'}
        • Weighted loss: {self.use_weighted_loss}
        • Learning rate: {self.learning_rate}
        • Number of classes: {self.num_classes}
        """

        return summary
