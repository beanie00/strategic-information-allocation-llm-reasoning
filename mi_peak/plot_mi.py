"""
MI Visualization: (1) Line plot (2) Word-shading HTML
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_line(df, title, output_file):
    """MI trajectory line plot"""
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(df['position'], df['mi_value'], 'b-', alpha=0.5, linewidth=0.8)
    scatter = ax.scatter(df['position'], df['mi_value'],
                         c=df['mi_value'], cmap='coolwarm',
                         s=15, vmin=df['mi_value'].min(), vmax=df['mi_value'].max())

    mean_val = df['mi_value'].mean()
    ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5,
               label=f'Mean: {mean_val:.2f}')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('MI Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='MI Value')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_comparison(df1, df2, title1, title2, output_file):
    """Side-by-side comparison plot of two trajectories"""
    from matplotlib.colors import LinearSegmentedColormap

    # Custom colormap: teal -> white -> pink
    colors = ['#06694D', '#FFFFFF', '#B33E7A']
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6))

    for ax, df, title in [(axes[0], df1, title1), (axes[1], df2, title2)]:
        ax.plot(df['position'], df['mi_value'], color='gray', alpha=0.3, linewidth=2)
        scatter = ax.scatter(df['position'], df['mi_value'],
                            c=df['mi_value'], cmap=cmap,
                            s=50, vmin=0, vmax=5)

        ax.set_ylabel('MI Value', fontsize=25)
        ax.set_title(title, fontsize=30)
        ax.set_xlim([0, 500])
        ax.tick_params(axis='both', labelsize=25)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5.5)

    axes[1].set_xlabel('Token Position', fontsize=25)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def mi_to_color(mi, mi_min, mi_max):
    """Convert MI value to color (teal -> white -> pink)"""
    norm = (mi - mi_min) / (mi_max - mi_min + 1e-6)
    norm = max(0, min(1, norm))

    # Teal rgb(52,181,145) -> white rgb(255,255,255) -> pink rgb(218,123,153)
    if norm < 0.5:
        t = norm * 2
        r = int(52 + t * (255 - 52))
        g = int(181 + t * (255 - 181))
        b = int(145 + t * (255 - 145))
    else:
        t = (norm - 0.5) * 2
        r = int(255 + t * (218 - 255))
        g = int(255 + t * (123 - 255))
        b = int(255 + t * (153 - 255))

    return f'rgb({r},{g},{b})'


def create_html_viz(df, title, output_file):
    """Generate word-shading HTML visualization"""
    mi_min, mi_max = df['mi_value'].min(), df['mi_value'].max()
    mi_mean = df['mi_value'].mean()

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Consolas', 'Monaco', monospace;
            padding: 20px;
            line-height: 2.2;
            max-width: 1400px;
            margin: 0 auto;
            background: #fafafa;
        }}
        .token {{
            display: inline;
            padding: 3px 2px;
            border-radius: 3px;
            cursor: pointer;
            margin: 1px;
            color: #333;
            font-weight: 900;
        }}
        .token:hover {{
            outline: 2px solid red;
            position: relative;
        }}
        .token:hover::after {{
            content: attr(data-info);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 100;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 8px;
        }}
        .legend-bar {{
            height: 25px;
            background: linear-gradient(to right, rgb(52,181,145), rgb(255,255,255), rgb(218,123,153));
            border-radius: 4px;
            margin: 10px 0;
        }}
        .content {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .newline {{ display: block; height: 10px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="legend">
        <strong>MI Legend:</strong> (hover for details)
        <div class="legend-bar"></div>
        <div style="display:flex; justify-content:space-between;">
            <span>Low ({mi_min:.2f})</span>
            <span>High ({mi_max:.2f})</span>
        </div>
        <div>Mean: {mi_mean:.3f} | Tokens: {len(df)}</div>
    </div>
    <div class="content">
"""

    for _, row in df.iterrows():
        token = str(row['token_str'])
        mi = row['mi_value']
        pos = row['position']
        color = mi_to_color(mi, mi_min, mi_max)

        token_display = token.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

        if '\\n' in token_display or '\n' in token_display:
            token_display = token_display.replace('\\n', '').replace('\n', '')
            html += f'<span class="token" style="background-color:{color}" data-info="pos:{pos} MI:{mi:.3f}">{token_display}</span><span class="newline"></span>'
        else:
            html += f'<span class="token" style="background-color:{color}" data-info="pos:{pos} MI:{mi:.3f}">{token_display}</span>'

    html += "</div></body></html>"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--csv2', type=str, default=None)
    parser.add_argument('--title', type=str, default='MI Trajectory')
    parser.add_argument('--title2', type=str, default='MI Trajectory 2')
    parser.add_argument('--output', type=str, default='mi_viz')
    args = parser.parse_args()

    df1 = pd.read_csv(args.csv)
    print(f"Loaded {len(df1)} tokens")

    if args.csv2 is None:
        plot_line(df1, args.title, f'{args.output}_line.png')
        create_html_viz(df1, args.title, f'{args.output}.html')
    else:
        df2 = pd.read_csv(args.csv2)
        plot_comparison(df1, df2, args.title, args.title2, f'{args.output}_comparison.png')
        create_html_viz(df1, args.title, f'{args.output}_1.html')
        create_html_viz(df2, args.title2, f'{args.output}_2.html')