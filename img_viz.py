import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Variáveis globais
current_volume = None
current_index = 1562 # testar index = 3710, 3703

current_text = ""
current_indices = None

# Botões para avançar e retroceder
btn_prev = widgets.Button(description="<< Anterior")
btn_next = widgets.Button(description="Próximo >>")

# Label para mostrar o índice atual
label_index = widgets.Label()

# Slider para escolher a slice
slice_slider = widgets.IntSlider(
    value=0, min=0, max=0, step=1, description='Slice:', continuous_update=False
)

# Output para imagem e informações
output = widgets.Output()

def load_volume_and_update(index):
    global current_volume, current_index, current_text, current_indices, current_series_data, current_vol_raw
    current_index = index
    uid = train_df.iloc[index]["SeriesInstanceUID"]
    mod = train_df.iloc[index]["Modality"]
    current_volume, current_indices, current_series_data = load_dicom(
        slice_size=config.slice_size,
        series_uid=uid,
        series_mapping_df=series_mapping_df,
        image_size=config.img_size,
        use_win_default=False
    )
    slice_slider.max = current_volume.shape[0] - 1
    slice_slider.value = 0
    label_index.value = f"Index atual: {current_index} / {len(train_df)-1} - Modality: {mod}"
    with output:
        clear_output(wait=True)
        show_slice(0)

def show_slice(slice_idx):
    plt.figure(figsize=(6, 6))
    plt.imshow(current_volume[slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx} / {current_volume.shape[0] - 1} | {current_text} | {current_indices}")
    plt.axis('off')
    plt.show()

def on_slice_change(change):
    if change['name'] == 'value' and change['type'] == 'change':
        with output:
            clear_output(wait=True)
            show_slice(change['new'])

def on_prev_clicked(b):
    global current_index
    if current_index > 0:
        load_volume_and_update(current_index - 1)

def on_next_clicked(b):
    global current_index
    if current_index < len(train_df) - 1:
        load_volume_and_update(current_index + 1)

# Liga eventos
slice_slider.observe(on_slice_change)
btn_prev.on_click(on_prev_clicked)
btn_next.on_click(on_next_clicked)

# Inicializa
load_volume_and_update(current_index)

# Layout
controls = widgets.HBox([btn_prev, label_index, btn_next])
display(widgets.VBox([controls, slice_slider, output]))
