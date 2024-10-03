# Description

The repository contains Python files for 4 proposed architectures of CNN to recongize emotions based on the GoEmotions dataset.

GoEmotions dataset: https://github.com/google-research/google-research/tree/master/goemotions  

# Comparison of 4 Proposed Architectures
<img src="https://github.com/user-attachments/assets/dcc838e4-c305-4fb7-82c6-6144f65cd9a3" alt="Model Metrics and Comparison" width="500"/>

## Architecture 1: Basic CNN Model
- **Layers:** Embedding → Conv1D → GlobalMaxPooling1D → Dense (64) → Dense (28 emotions, sigmoid)
- **Performance:**
  - Test Accuracy: 0.3685
  - Test Precision: 0.6074
  - Test Recall: 0.4253
  - Test F1 Score: 0.5003
- **Key Observations:**
  - This model provides a baseline with a simple structure, having one Conv1D layer.
  - The model achieves decent performance, but its recall is relatively low, indicating difficulty in identifying all relevant emotions.
  - Precision is moderate, but overall accuracy and F1 score suggest room for improvement.

## Architecture 2: CNN with Two Conv1D Layers
- **Layers:** Embedding → Conv1D → Conv1D → GlobalMaxPooling1D → Dense (64) → Dense (28 emotions, sigmoid)
- **Performance:**
  - Test Accuracy: 0.3923
  - Test Precision: 0.6068
  - Test Recall: 0.4375
  - Test F1 Score: 0.5084
- **Key Observations:**
  - Adding an extra Conv1D layer improved both test accuracy and recall, showing better emotion detection.
  - The slight increase in F1 score reflects better balance between precision and recall, indicating that additional layers help capture intricate features.
  - Precision remains similar, suggesting correct classifications without being more conservative.

## Architecture 3: CNN with L2 Regularization and Batch Normalization
- **Layers:** Embedding → Conv1D (with L2 regularization) → BatchNormalization → GlobalMaxPooling1D → Dense (64 with L2 regularization) → Dense (28 emotions, sigmoid)
- **Performance:**
  - Test Accuracy: 0.3683
  - Test Precision: 0.6891
  - Test Recall: 0.3772
  - Test F1 Score: 0.4875
- **Key Observations:**
  - Despite introducing L2 regularization and batch normalization, test accuracy remains similar to Architecture 1.
  - Precision increased significantly (from 0.607 to 0.689), indicating fewer false positives but lower recall (0.377), showing caution in identifying emotions.
  - The lower F1 score compared to Architecture 2 suggests a trade-off between precision and recall.
  - Regularization techniques can reduce overfitting but may need careful tuning to avoid reduced recall.

## Architecture 4: CNN with Two Conv1D Layers and Larger Batch Size
- **Layers:** Embedding → Conv1D → Conv1D → GlobalMaxPooling1D → Dense (64) → Dense (28 emotions, sigmoid)
- **Performance:**
  - Test Accuracy: 0.4008
  - Test Precision: 0.6570
  - Test Recall: 0.4165
  - Test F1 Score: 0.5098
- **Key Observations:**
  - Similar architecture to Architecture 2 but trained with a larger batch size (64 instead of 32).
  - Test accuracy and F1 score are slightly higher than Architecture 2, indicating better convergence.
  - Precision is higher (0.657 vs. 0.606), but recall is slightly lower (0.416 vs. 0.437), reflecting improved emotion identification while maintaining precision.
  - The larger batch size likely improved training efficiency.

## Conclusion
- **Best Architecture:** Architecture 4 achieves the best balance of test accuracy (0.4008) and F1 score (0.5098), indicating that deeper architectures combined with larger batch sizes improve performance. Architecture 2 is a close second, providing better recall.
- **Regularization:** L2 regularization and batch normalization (as in Architecture 3) significantly improve precision but may compromise recall, which is crucial for multi-label emotion classification.
- **Extra Conv1D Layers:** The addition of a second Conv1D layer (Architectures 2 and 4) consistently improves performance across all metrics compared to the single Conv1D architecture (Architecture 1).
- **Batch Size Impact:** Architecture 4 shows that increasing the batch size to 64 improves both accuracy and precision while slightly reducing recall.
