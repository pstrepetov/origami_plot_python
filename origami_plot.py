import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ALPHA = 0.25

def calculate_polygon_area(x, y):
    """
    Calculate the area of a polygon given its vertices.
    
    Parameters:
        x (list or array-like): x-coordinates of the polygon vertices.
        y (list or array-like): y-coordinates of the polygon vertices.
    
    Returns:
        float: Area of the polygon.
    """
    x = np.array(x)
    y = np.array(y)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def plot_multiple_stars_with_spider_net(outer_radii_list, inner_radius=0.1, 
                                         color_list=None, linewidth=0.5, linestyle_list=None,
                                         fill=True, fill_color_list=None, alpha_list=None,
                                         spider_net_radii=None, spider_net_color='gray', spider_net_alpha=0.3,
                                         vertex_names=None, vertex_fontsize=10, vertex_fontcolor='black',
                                         star_labels=None, figsize=(8,8)):
    """
    Plots multiple stars with custom outer radii for each star.
    Adds a spider net background with concentric polygons at specified radii.
    Optionally adds vertex names, a legend with star areas, and spider net radius labels to the plot.
    
    Parameters:
        outer_radii_list (list of lists or array-like): List of outer radii for each star. Each element must have the same length.
        inner_radius (float): Radius of the inner points of the stars.
        color_list (list of str): Colors of the star outlines. Must match the number of stars.
        linewidth (float): Width of the star outlines.
        linestyle_list (list of str): Line styles for the star outlines. Must match the number of stars.
        fill (bool): Whether to fill the stars with color.
        fill_color_list (list of str): Fill colors for the stars. Must match the number of stars.
        alpha_list (list of float): Transparency levels for the stars. Must match the number of stars.
        spider_net_radii (list or array-like): Radii for the spider net background polygons.
        spider_net_color (str): Color of the spider net lines.
        spider_net_alpha (float): Transparency level of the spider_net lines.
        vertex_names (list or array-like): Labels for each vertex. Must have length N.
        vertex_fontsize (int): Font size for vertex names.
        vertex_fontcolor (str): Font color for vertex names.
        star_labels (list of str): Labels for each star in the legend. Must match the number of stars.
        figsize (tuple): The width and height of the plot in inches.
    """
    # Infer N (number of vertices) from the first star's outer radii
    N = len(outer_radii_list[0])
    if N < 3:
        raise ValueError("The number of vertices (N) must be at least 3.")
    for i, outer_radii in enumerate(outer_radii_list):
        if len(outer_radii) != N:
            raise ValueError(f"Length of outer_radii_list[{i}] ({len(outer_radii)}) does not match the number of vertices (N={N}).")
    num_stars = len(outer_radii_list)
    
    # Default values for lists
    if color_list is None:
        color_list = plt.cm.tab10.colors[:num_stars]  # Use distinct colors from colormap
    if fill_color_list is None:
        fill_color_list = plt.cm.tab10.colors[:num_stars]
    if alpha_list is None:
        alpha_list = [DEFAULT_ALPHA] * num_stars
    if linestyle_list is None:
        linestyle_list = ['-' for _ in range(num_stars)]  # Default solid line
    
    # Validate lengths of lists
    if len(color_list) != num_stars:
        raise ValueError(f"Length of color_list ({len(color_list)}) must match the number of stars ({num_stars}).")
    if len(fill_color_list) != num_stars:
        raise ValueError(f"Length of fill_color_list ({len(fill_color_list)}) must match the number of stars ({num_stars}).")
    if len(alpha_list) != num_stars:
        raise ValueError(f"Length of alpha_list ({len(alpha_list)}) must match the number of stars ({num_stars}).")
    if len(linestyle_list) != num_stars:
        raise ValueError(f"Length of linestyle_list ({len(linestyle_list)}) must match the number of stars ({num_stars}).")
    if star_labels is not None and len(star_labels) != num_stars:
        raise ValueError(f"Length of star_labels ({len(star_labels)}) must match the number of stars ({num_stars}).")
    
    # Generate angles for the star points
    angles = np.linspace(0, 2 * np.pi, 2 * N, endpoint=False)
    
    # Plot the spider net background
    fig, ax = plt.subplots(figsize=figsize)
    if spider_net_radii is not None:
        for radius in spider_net_radii:
            x_polygon = radius * np.cos(angles[::2])
            y_polygon = radius * np.sin(angles[::2])
            ax.plot(np.append(x_polygon, x_polygon[0]), np.append(y_polygon, y_polygon[0]),
                    color=spider_net_color, linewidth=1, alpha=spider_net_alpha)
        
        # Add radial lines from the center to the vertices
        for angle in angles[::2]:
            ax.plot([0, max(spider_net_radii) * np.cos(angle)],
                    [0, max(spider_net_radii) * np.sin(angle)],
                    color=spider_net_color, linewidth=1, alpha=spider_net_alpha)
        
        # Add spider net radius labels between 1st and 2nd radial lines
        label_angle = angles[1]  # Angle for the radial line where labels will be placed
        for radius in spider_net_radii[:-1]:
            label_x = radius * np.cos(label_angle) ** 2
            label_y = radius * np.sin(label_angle) * np.cos(label_angle)
            ax.text(label_x, label_y, f"{radius:.1f}", fontsize=8, color=spider_net_color,
                    ha='center', va='center')
    
    # Plot each star and calculate its area
    legend_handles = []
    for i, outer_radii in enumerate(outer_radii_list):
        # Alternate between outer and inner points for the star
        radii = []
        for j in range(2 * N):
            if j % 2 == 0:  # Outer point
                radii.append(outer_radii[j // 2])
            else:  # Inner point
                radii.append(inner_radius)
        
        # Compute the x and y coordinates of the star points
        x_star = np.array(radii) * np.cos(angles)
        y_star = np.array(radii) * np.sin(angles)
        
        # Close the star by repeating the first point
        x_star = np.append(x_star, x_star[0])
        y_star = np.append(y_star, y_star[0])
        
        # Calculate the area of the star
        area = calculate_polygon_area(x_star, y_star)
        
        # Create label for the legend
        if star_labels:
            label = f"{star_labels[i]} (Area: {area:.2f})"
        else:
            label = f"Star {i+1} (Area: {area:.2f})"
        
        # Plot the star
        if fill:
            star_patch = ax.fill(x_star, y_star, color=fill_color_list[i], edgecolor=color_list[i], 
                                 linewidth=linewidth, linestyle=linestyle_list[i], alpha=alpha_list[i],
                                 label=label)[0]
        else:
            star_patch = ax.plot(x_star, y_star, color=color_list[i], linewidth=linewidth, 
                                 linestyle=linestyle_list[i], alpha=alpha_list[i],
                                 label=label)[0]
        
        # Add to legend handles
        legend_handles.append(star_patch)
    
    # Add vertex names
    if vertex_names is not None:
        for i, name in enumerate(vertex_names):
            x_label = max(spider_net_radii) * np.cos(angles[::2][i]) * 1.1  # Offset slightly outward
            y_label = max(spider_net_radii) * np.sin(angles[::2][i]) * 1.1  # Offset slightly outward
            ax.text(x_label, y_label, name, fontsize=vertex_fontsize, color=vertex_fontcolor,
                    ha='center', va='center')
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10, frameon=True, title="Stars")
    
    # Set aspect ratio and limits
    max_radius = max([max(outer_radii + [inner_radius]) for outer_radii in outer_radii_list] + 
                     (spider_net_radii if spider_net_radii else []))
    ax.set_aspect('equal')
    ax.set_xlim(-max_radius * 1.2, max_radius * 1.2)
    ax.set_ylim(-max_radius * 1.2, max_radius * 1.2)
    ax.axis('off')  # Hide axes
    
    plt.show()
)
