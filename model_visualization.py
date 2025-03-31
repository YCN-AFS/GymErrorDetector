import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import seaborn as sns

def draw_lstm_architecture(save_path='lstm_architecture.png'):
    """Vẽ kiến trúc mô hình LSTM"""
    fig, ax = plt.subplots(figsize=(15, 8))  # Tạo figure và axes
    
    # Bỏ style, thêm grid thủ công
    ax.grid(True, alpha=0.3)
    colors = sns.color_palette("husl", 8)
    
    # Vẽ input layer
    input_color = colors[0]
    ax.plot([1, 1], [0, 4], 'k-', linewidth=2)
    for i in range(5):
        circle = Circle((1, i*0.8), 0.1, fc=input_color)
        ax.add_patch(circle)  # Sử dụng ax.add_patch thay vì plt.add_patch
    ax.text(1, 4.3, 'Input Layer\n(132 features)', ha='center')
    
    # Vẽ LSTM layer
    lstm_color = colors[1]
    x_lstm = 3
    for i in range(4):
        rect = Rectangle((x_lstm-0.5, i*0.8-0.2), 1, 0.4,
                        fc=lstm_color, alpha=0.3)
        ax.add_patch(rect)  # Sử dụng ax.add_patch
        ax.text(x_lstm, i*0.8, 'LSTM', ha='center', va='center')
    ax.text(x_lstm, 4.3, 'LSTM Layer\n(128 units)', ha='center')
    
    # Vẽ Batch Normalization
    bn_color = colors[2]
    x_bn = 5
    ax.plot([x_bn, x_bn], [0, 3.2], 'k-', linewidth=2)
    ax.text(x_bn, 4.3, 'Batch Norm\n(128)', ha='center')
    
    # Vẽ Dropout
    dropout_color = colors[3]
    x_dropout = 7
    ax.plot([x_dropout, x_dropout], [0, 3.2], 'k--', linewidth=2)
    ax.text(x_dropout, 4.3, 'Dropout\n(0.5)', ha='center')
    
    # Vẽ Dense layers
    dense_colors = [colors[4], colors[5], colors[6]]
    x_dense = [9, 11, 13]
    dense_sizes = [64, 32, 1]
    
    for i, (x, size, color) in enumerate(zip(x_dense, dense_sizes, dense_colors)):
        ax.plot([x, x], [1.5, 2.5], 'k-', linewidth=2)
        rect = Rectangle((x-0.3, 1.8), 0.6, 0.4,
                        fc=color, alpha=0.3)
        ax.add_patch(rect)  # Sử dụng ax.add_patch
        ax.text(x, 4.3, f'Dense Layer\n({size} units)', ha='center')
    
    # Vẽ các mũi tên kết nối
    arrows = [(1.2, 3), (3.6, 5), (5.2, 7), (7.2, 9), (9.2, 11), (11.2, 13)]
    for start, end in arrows:
        arrow = FancyArrowPatch((start, 2), (end, 2),
                               arrowstyle='->',
                               mutation_scale=20)
        ax.add_patch(arrow)
    
    # Thêm các chú thích
    ax.text(13, 1.5, 'Output\n(Sigmoid)', ha='center')
    
    # Thiết lập trục
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Thêm tiêu đề
    ax.set_title('LSTM Model Architecture for Pose Assessment', pad=20, size=14)
    
    # Thêm chú thích
    ax.text(0.02, 0.02, 
                'Input: Pose landmarks sequence\nOutput: Movement quality score (0-1)',
                ha='left', fontsize=10)
    
    # Lưu hình
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_cnn_architecture(save_path='cnn_architecture.png'):
    """Vẽ kiến trúc mô hình CNN (nếu sử dụng)"""
    plt.figure(figsize=(15, 8))
    
    # Bỏ style, thêm grid thủ công
    plt.grid(True, alpha=0.3)
    colors = sns.color_palette("husl", 8)
    
    # Vẽ input layer (hình ảnh)
    input_color = colors[0]
    x_input = 1
    rect = Rectangle((x_input-0.5, 1), 1, 2,
                    fc=input_color, alpha=0.3)
    plt.gca().add_patch(rect)
    plt.text(x_input, 4.3, 'Input Image\n(480x640x3)', ha='center')
    
    # Vẽ các Conv layers
    conv_color = colors[1]
    x_conv = [3, 5]
    for i, x in enumerate(x_conv):
        rect = Rectangle((x-0.4, 1.2), 0.8, 1.6,
                        fc=conv_color, alpha=0.3)
        plt.gca().add_patch(rect)
        plt.text(x, 4.3, f'Conv2D\n(64 filters)', ha='center')
    
    # Vẽ MaxPooling
    pool_color = colors[2]
    x_pool = 7
    rect = Rectangle((x_pool-0.3, 1.4), 0.6, 1.2,
                    fc=pool_color, alpha=0.3)
    plt.gca().add_patch(rect)
    plt.text(x_pool, 4.3, 'MaxPooling', ha='center')
    
    # Vẽ Flatten
    flat_color = colors[3]
    x_flat = 9
    plt.plot([x_flat, x_flat], [1.5, 2.5], 'k-', linewidth=2)
    plt.text(x_flat, 4.3, 'Flatten', ha='center')
    
    # Vẽ Dense layers
    dense_colors = [colors[4], colors[5]]
    x_dense = [11, 13]
    dense_sizes = [128, 1]
    
    for i, (x, size, color) in enumerate(zip(x_dense, dense_sizes, dense_colors)):
        plt.plot([x, x], [1.5, 2.5], 'k-', linewidth=2)
        rect = Rectangle((x-0.3, 1.8), 0.6, 0.4,
                        fc=color, alpha=0.3)
        plt.gca().add_patch(rect)
        plt.text(x, 4.3, f'Dense\n({size})', ha='center')
    
    # Vẽ các mũi tên kết nối
    arrows = [(1.6, 3), (3.6, 5), (5.6, 7), (7.6, 9), (9.6, 11), (11.6, 13)]
    for start, end in arrows:
        arrow = FancyArrowPatch((start, 2), (end, 2),
                               arrowstyle='->',
                               mutation_scale=20)
        plt.gca().add_patch(arrow)
    
    # Thiết lập trục
    plt.xlim(0, 14)
    plt.ylim(0, 5)
    plt.axis('off')
    
    # Thêm tiêu đề
    plt.title('CNN Model Architecture for Pose Assessment', pad=20, size=14)
    
    # Thêm chú thích
    plt.figtext(0.02, 0.02, 
                'Input: RGB image\nOutput: Movement quality score (0-1)',
                ha='left', fontsize=10)
    
    # Lưu hình
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Vẽ cả hai kiến trúc
    draw_lstm_architecture('lstm_architecture.png')
    draw_cnn_architecture('cnn_architecture.png')
    print("Đã tạo hình minh họa kiến trúc mô hình!") 