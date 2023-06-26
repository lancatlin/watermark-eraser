from matplotlib import pyplot as plt


def plot_3d(df, clustering=None):
    # 從 DataFrame 中取得 mixed 和 count 資料
    mixed_values = df['mixed']
    count_values = df['count']

    # 將 mixed 的 RGB 值分開成三個獨立的陣列
    mixed_b = [mixed[0] for mixed in mixed_values]
    mixed_g = [mixed[1] for mixed in mixed_values]
    mixed_r = [mixed[2] for mixed in mixed_values]

    # 繪製散點圖
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 繪製散點圖
    ax.scatter3D(mixed_b, mixed_g, mixed_r, s=5 * count_values, c=clustering, cmap='viridis')

    # 設定標籤和標題
    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    plt.title('3D Scatter Plot')

