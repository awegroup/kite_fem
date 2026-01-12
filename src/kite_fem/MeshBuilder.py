#!/usr/bin/env python3
"""
Interactive Mesh Builder for Kite FEM

Click on points to create connections:
- LEFT CLICK: Select points for connections
- '1' key + 2 points: Create spring connection
- '2' key + 2 points: Create beam connection  
- '3' key + 3 points: Create pulley connection
- 'c' key: Clear all selections
- 'u' key: Undo last connection
- 'r' key: Reset all connections
- 'q' key: Quit and print matrices
- 'h' key: Show help

The tool will display the matrices when you quit.
"""

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox

class InteractiveMeshBuilder:
    def __init__(self, initial_conditions=None):
        # Handle optional initial conditions
        if initial_conditions is None:
            self.initial_conditions = []
            self.points = np.empty((0, 3))  # Empty array with 3 columns
        else:
            self.initial_conditions = initial_conditions
            self.points = np.array([ic[0] for ic in initial_conditions])
        
        self.num_points = len(self.points)
        
        # Connection storage
        self.spring_matrix = []
        self.beam_matrix = []
        self.pulley_matrix = []
        
        # Selection state
        self.selected_points = []
        self.mode = 'select'  # 'select', 'spring', 'beam', 'pulley', 'add_point'
        
        # Default parameters
        self.spring_k = 50000.0
        self.spring_c = 0.9
        self.spring_type = 'default'
        self.beam_diameter = 0.1
        self.beam_pressure = 0.3
        self.pulley_k = 5000.0
        self.pulley_c = 0.9
        
        # Graphics storage
        self.connection_lines = []
        
        self.setup_plot()
        self.setup_callbacks()
    
    def get_spring_properties(self, n1, n2, current_length):
        """Get spring properties from user input dialog"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Create dialog
        dialog_title = f"Spring Properties (nodes {n1}-{n2})"
        
        # Get stiffness
        k_input = simpledialog.askstring(
            dialog_title, 
            f"Enter spring stiffness k (current: {self.spring_k}):",
            initialvalue=str(self.spring_k)
        )
        if k_input is None:  # User cancelled
            root.destroy()
            return None
            
        # Get rest length
        l0_input = simpledialog.askstring(
            dialog_title,
            f"Enter rest length l0 (current length: {current_length:.3f}, leave empty to use current):",
            initialvalue=""
        )
        if l0_input is None:
            root.destroy()
            return None
        
        # Get spring type with dropdown
        spring_type = self.get_spring_type_choice(dialog_title)
        if spring_type is None:
            root.destroy()
            return None
        
        root.destroy()
        
        # Process inputs
        l0 = current_length if l0_input.strip() == "" else l0_input.strip()
        
        return {
            'k': k_input.strip(),
            'l0': l0,
            'type': spring_type
        }
    
    def get_spring_type_choice(self, parent_title):
        """Show a dialog with spring type choices"""
        # Create a new window for the choice
        choice_window = tk.Toplevel()
        choice_window.title(parent_title + " - Spring Type")
        choice_window.geometry("300x200")
        choice_window.resizable(False, False)
        
        # Center the window
        choice_window.transient()
        choice_window.grab_set()
        
        result = {"value": self.spring_type}  # Use dict to store result
        
        # Add label
        label = tk.Label(choice_window, text="Select spring type:", font=("Arial", 12))
        label.pack(pady=10)
        
        # Create variable for radio buttons
        selected_type = tk.StringVar(value=self.spring_type)
        
        # Add radio buttons
        radio1 = tk.Radiobutton(choice_window, text="default", variable=selected_type, 
                               value="default", font=("Arial", 11))
        radio1.pack(pady=5)
        
        radio2 = tk.Radiobutton(choice_window, text="noncompressive", variable=selected_type, 
                               value="noncompressive", font=("Arial", 11))
        radio2.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(choice_window)
        button_frame.pack(pady=15)
        
        def on_ok():
            result["value"] = selected_type.get()
            choice_window.destroy()
            
        def on_cancel():
            choice_window.destroy()
        
        # Add buttons
        ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=8)
        ok_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=8)
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Wait for window to close
        choice_window.wait_window()
        
        # Return None if cancelled, otherwise return the selected value
        return result.get("value") if "value" in result else None
    
    def get_fixed_status_choice(self, parent_title):
        """Show a dialog with fixed status choices (True/False)"""
        # Create a new window for the choice
        choice_window = tk.Toplevel()
        choice_window.title(parent_title + " - Fixed Status")
        choice_window.geometry("300x200")
        choice_window.resizable(False, False)
        
        # Center the window
        choice_window.transient()
        choice_window.grab_set()
        
        result = {"value": False}  # Default to False (not fixed)
        
        # Add label
        label = tk.Label(choice_window, text="Is this point fixed in space?", font=("Arial", 12))
        label.pack(pady=10)
        
        # Create variable for radio buttons
        selected_fixed = tk.BooleanVar(value=False)
        
        # Add radio buttons
        radio1 = tk.Radiobutton(choice_window, text="False (movable)", variable=selected_fixed, 
                               value=False, font=("Arial", 11))
        radio1.pack(pady=5)
        
        radio2 = tk.Radiobutton(choice_window, text="True (fixed)", variable=selected_fixed, 
                               value=True, font=("Arial", 11))
        radio2.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(choice_window)
        button_frame.pack(pady=15)
        
        def on_ok():
            result["value"] = selected_fixed.get()
            choice_window.destroy()
            
        def on_cancel():
            choice_window.destroy()
        
        # Add buttons
        ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=8)
        ok_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=8)
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Wait for window to be closed
        choice_window.wait_window()
        
        return result["value"]
    
    def get_beam_properties(self, n1, n2):
        """Get beam properties from user input dialog"""
        root = tk.Tk()
        root.withdraw()
        
        dialog_title = f"Beam Properties (nodes {n1}-{n2})"
        
        # Get diameter
        d_input = simpledialog.askstring(
            dialog_title,
            f"Enter beam diameter d (current: {self.beam_diameter}):",
            initialvalue=str(self.beam_diameter)
        )
        if d_input is None:
            root.destroy()
            return None
            
        # Get pressure
        p_input = simpledialog.askstring(
            dialog_title,
            f"Enter beam pressure p (current: {self.beam_pressure}):",
            initialvalue=str(self.beam_pressure)
        )
        if p_input is None:
            root.destroy()
            return None
        
        root.destroy()
        
        return {
            'd': d_input.strip(),
            'p': p_input.strip()
        }
    
    def get_pulley_properties(self, n1, n2, n3, current_length):
        """Get pulley properties from user input dialog"""
        root = tk.Tk()
        root.withdraw()
        
        dialog_title = f"Pulley Properties (nodes {n1}-{n2}-{n3})"
        
        # Get stiffness
        k_input = simpledialog.askstring(
            dialog_title,
            f"Enter pulley stiffness k (current: {self.pulley_k}):",
            initialvalue=str(self.pulley_k)
        )
        if k_input is None:
            root.destroy()
            return None
            
        # Get rest length
        l0_input = simpledialog.askstring(
            dialog_title,
            f"Enter rest length l0 (current length: {current_length:.3f}, leave empty to use current):",
            initialvalue=""
        )
        if l0_input is None:
            root.destroy()
            return None
        
        root.destroy()
        
        # Process inputs
        l0 = current_length if l0_input.strip() == "" else l0_input.strip()
        
        return {
            'k': k_input.strip(),
            'l0': l0
        }
        
    def set_equal_axes(self):
        """Make axes equal using code from Plotting.py"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        zmid = (zlim[0] + zlim[1]) / 2
        maximum = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
        self.ax.set_xlim([xmid - maximum / 2, xmid + maximum / 2])
        self.ax.set_ylim([ymid - maximum / 2, ymid + maximum / 2])
        self.ax.set_zlim([zmid - maximum / 2, zmid + maximum / 2])
        self.ax.set_box_aspect([1, 1, 1])
        
    def setup_plot(self):
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot all points if any exist
        if len(self.points) > 0:
            self.point_scatter = self.ax.scatter(
                self.points[:, 0], self.points[:, 1], self.points[:, 2],
                c='blue', s=80, picker=True, alpha=0.8, edgecolors='black'
            )
            
            # Add point labels
            for i, point in enumerate(self.points):
                self.ax.text(point[0], point[1], point[2], f'{i}', fontsize=9, color='black')
        else:
            # Create empty scatter plot for later updates
            self.point_scatter = self.ax.scatter([], [], [], 
                c='blue', s=80, picker=True, alpha=0.8, edgecolors='black')
            
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')  
        self.ax.set_zlabel('Z')
        
        # Set default axes limits if no points
        if len(self.points) == 0:
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([-5, 5])
            self.ax.set_zlim([-5, 5])
        else:
            # Make axes equal (from Plotting.py)
            self.set_equal_axes()
        
        self.update_title()
        
        # Add instruction text
        self.info_text = self.fig.text(0.02, 0.98, self.get_instructions(), 
                                      verticalalignment='top', fontsize=10, 
                                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
    def get_instructions(self):
        selected_str = ", ".join(map(str, self.selected_points)) if self.selected_points else "None"
        return (f"Mode: {self.mode.upper()}\n" +
                f"Selected: {selected_str}\n" +
                f"Points: {len(self.points)} | " +
                f"Springs: {len(self.spring_matrix)} | " +
                f"Beams: {len(self.beam_matrix)} | " +
                f"Pulleys: {len(self.pulley_matrix)}\n\n" +
                "Controls:\n" +
                "a=add point, 1=spring, 2=beam, 3=pulley, 4=select\n" +
                "c=clear, u=undo, r=reset, q=quit, h=help")
        
    def update_title(self):
        self.ax.set_title(f'Interactive Mesh Builder - Mode: {self.mode.upper()}\n' + 
                         f'Total Connections: {len(self.spring_matrix) + len(self.beam_matrix) + len(self.pulley_matrix)}')
        
    def setup_callbacks(self):
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_pick(self, event):
        if event.artist != self.point_scatter:
            return
            
        # Get the index of the picked point
        ind = event.ind[0] if len(event.ind) > 0 else None
        if ind is None:
            return
            
        if ind in self.selected_points:
            # Deselect if already selected
            self.selected_points.remove(ind)
            print(f"Deselected point {ind}")
        else:
            # Select point
            self.selected_points.append(ind)
            print(f"Selected point {ind} at {self.points[ind]}")
            
        # Check if we have enough points for current mode
        if self.mode == 'spring' and len(self.selected_points) == 2:
            self.create_spring()
        elif self.mode == 'beam' and len(self.selected_points) == 2:
            self.create_beam()
        elif self.mode == 'pulley' and len(self.selected_points) == 3:
            self.create_pulley()
            
        self.update_display()
        
    def on_key(self, event):
        print(f"Key pressed: {event.key}")
        
        if event.key == '1':
            self.mode = 'spring'
            self.selected_points = []
            print("Switched to SPRING mode. Click 2 points to create a spring.")
        elif event.key == '2':
            self.mode = 'beam'
            self.selected_points = []
            print("Switched to BEAM mode. Click 2 points to create a beam.")
        elif event.key == '3':
            self.mode = 'pulley'
            self.selected_points = []
            print("Switched to PULLEY mode. Click 3 points to create a pulley.")
        elif event.key == '4':
            self.mode = 'select'
            self.selected_points = []
            print("Switched to SELECT mode. Click points to select/deselect them.")
        elif event.key == 'a':
            self.mode = 'add_point'
            self.selected_points = []
            print("Switched to ADD POINT mode. Click on empty space to add a point.")
        elif event.key == 'c':
            self.selected_points = []
            print("Cleared selection.")
        elif event.key == 'u':
            self.undo_last()
        elif event.key == 'r':
            self.reset_all()
        elif event.key == 'q':
            self.quit_and_print()
            return
        elif event.key == 'h':
            self.show_help()
        elif event.key == 'e':
            self.show_parameters()
            
        self.update_display()
        
    def on_close(self, event):
        """Handle window close event by printing results"""
        print("\n" + "="*80)
        print("WINDOW CLOSED - PRINTING RESULTS")
        print("="*80)
        self.print_matrices()
        
    def on_click(self, event):
        """Handle mouse clicks for adding points in add_point mode"""
        if self.mode != 'add_point' or event.inaxes != self.ax:
            return
            
        # Get coordinates from dialog (ignore click position)
        coords_data = self.get_point_coordinates()
        if coords_data is None:
            return
            
        self.add_point(coords_data['coordinates'], coords_data['fixed'])
    
    def get_point_coordinates(self):
        """Get point coordinates and fixed status from user input dialog"""
        root = tk.Tk()
        root.withdraw()
        
        dialog_title = "New Point Coordinates"
        
        # Get X coordinate
        x_input = simpledialog.askstring(
            dialog_title,
            "Enter X coordinate:",
            initialvalue="0.0"
        )
        if x_input is None:
            root.destroy()
            return None
            
        # Get Y coordinate
        y_input = simpledialog.askstring(
            dialog_title,
            "Enter Y coordinate:",
            initialvalue="0.0"
        )
        if y_input is None:
            root.destroy()
            return None
            
        # Get Z coordinate
        z_input = simpledialog.askstring(
            dialog_title,
            "Enter Z coordinate:",
            initialvalue="0.0"
        )
        if z_input is None:
            root.destroy()
            return None
        
        # Get fixed status
        fixed_status = self.get_fixed_status_choice(dialog_title)
        if fixed_status is None:
            root.destroy()
            return None
        
        root.destroy()
        
        try:
            x = float(x_input.strip())
            y = float(y_input.strip())
            z = float(z_input.strip())
            return {'coordinates': [x, y, z], 'fixed': fixed_status}
        except ValueError:
            print("Invalid coordinates entered!")
            return None
    
    def add_point(self, coordinates, fixed=False):
        """Add a new point to the mesh"""
        # Create new initial condition entry
        # Format: [position, velocity, mass, fixed]
        new_ic = [coordinates, [0.0, 0.0, 0.0], 1.0, fixed]
        
        # Add to initial conditions
        self.initial_conditions.append(new_ic)
        
        # Update points array
        if len(self.points) == 0:
            self.points = np.array([coordinates])
        else:
            self.points = np.vstack([self.points, coordinates])
            
        self.num_points = len(self.points)
        
        fixed_status = "fixed" if fixed else "movable"
        print(f"✓ Added point {self.num_points-1} at {coordinates} ({fixed_status})")
        
        # Update axes size to accommodate new point
        self.update_axes_size()
        
        # Update the display
        self.update_display()
    
    def update_axes_size(self):
        """Update axes limits to accommodate all points with some padding"""
        if len(self.points) == 0:
            return
            
        # Get min/max coordinates
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        
        # Add padding (20% of range)
        ranges = max_coords - min_coords
        padding = np.maximum(ranges * 0.2, 1.0)  # Minimum padding of 1.0
        
        # Set new limits
        self.ax.set_xlim(min_coords[0] - padding[0], max_coords[0] + padding[0])
        self.ax.set_ylim(min_coords[1] - padding[1], max_coords[1] + padding[1])
        self.ax.set_zlim(min_coords[2] - padding[2], max_coords[2] + padding[2])
        
        # Keep equal aspect ratio
        self.set_equal_axes()
        
    def print_matrices(self):
        """Helper method to print the matrices without closing the plot"""
        print(f"\n# Initial Conditions Matrix ({len(self.initial_conditions)} points)")
        print("# Format: [position, velocity, mass, fixed]")
        if self.initial_conditions:
            print("initial_conditions = [")
            for ic in self.initial_conditions:
                print(f"    {ic},")
            print("]")
        else:
            print("initial_conditions = []")
        
        print(f"\n# Spring Matrix ({len(self.spring_matrix)} connections)")
        print("# Format: [n1, n2, k, c, l0, springtype]")
        if self.spring_matrix:
            print("spring_matrix = [")
            for spring in self.spring_matrix:
                print(f"    {spring},")
            print("]")
        else:
            print("spring_matrix = []")
        
        print(f"\n# Beam Matrix ({len(self.beam_matrix)} connections)")
        print("# Format: [n1, n2, d, p]")
        if self.beam_matrix:
            print("beam_matrix = [")
            for beam in self.beam_matrix:
                print(f"    {beam},")
            print("]")
        else:
            print("beam_matrix = []")
        
        print(f"\n# Pulley Matrix ({len(self.pulley_matrix)} connections)")
        print("# Format: [n1, n2, n3, k, c, l0]")
        if self.pulley_matrix:
            print("pulley_matrix = [")
            for pulley in self.pulley_matrix:
                print(f"    {pulley},")
            print("]")
        else:
            print("pulley_matrix = []")
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print(f"Points: {len(self.initial_conditions)}")
        print(f"Springs: {len(self.spring_matrix)}")
        print(f"Beams: {len(self.beam_matrix)}")  
        print(f"Pulleys: {len(self.pulley_matrix)}")
        print(f"Total connections: {len(self.spring_matrix) + len(self.beam_matrix) + len(self.pulley_matrix)}")
        print("="*80)
        print("USAGE:")
        print("Copy the matrices above into your FEM code like this:")
        print("kite = FEM_structure(initial_conditions, spring_matrix=spring_matrix,")
        print("                     beam_matrix=beam_matrix, pulley_matrix=pulley_matrix)")
        print("="*80)
        
    def create_spring(self):
        if len(self.selected_points) != 2:
            return
            
        n1, n2 = self.selected_points
        # Calculate current length as distance between points
        p1, p2 = self.points[n1], self.points[n2]
        current_length = np.linalg.norm(p2 - p1)
        
        # Get properties from user dialog
        properties = self.get_spring_properties(n1, n2, current_length)
        if properties is None:  # User cancelled
            self.selected_points = []
            return
        
        # Use current length if l0 is empty, otherwise use provided value
        l0 = current_length if properties['l0'] == current_length else properties['l0']
        
        spring_conn = [n1, n2, properties['k'], 0,  l0, properties['type']]
        self.spring_matrix.append(spring_conn)
        
        # Add visual line
        line = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                           'g-', linewidth=2, alpha=0.8, label='Spring')[0]
        self.connection_lines.append(('spring', line))
        
        self.selected_points = []
        print(f"✓ Spring created: nodes {n1}-{n2}, k={properties['k']}, c={0}, l0={l0}, type={properties['type']}")
        
    def create_beam(self):
        if len(self.selected_points) != 2:
            return
            
        n1, n2 = self.selected_points
        
        # Get properties from user dialog
        properties = self.get_beam_properties(n1, n2)
        if properties is None:  # User cancelled
            self.selected_points = []
            return
        
        beam_conn = [n1, n2, properties['d'], properties['p']]
        self.beam_matrix.append(beam_conn)
        
        # Add visual line
        p1, p2 = self.points[n1], self.points[n2]
        line = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                           'b-', linewidth=5, alpha=0.8, label='Beam')[0]
        self.connection_lines.append(('beam', line))
        
        self.selected_points = []
        print(f"✓ Beam created: nodes {n1}-{n2}, d={properties['d']}, p={properties['p']}")
        
    def create_pulley(self):
        if len(self.selected_points) != 3:
            return
            
        n1, n2, n3 = self.selected_points
        # Calculate total current length for pulley
        p1, p2, p3 = self.points[n1], self.points[n2], self.points[n3]
        current_length = np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
        
        # Get properties from user dialog
        properties = self.get_pulley_properties(n1, n2, n3, current_length)
        if properties is None:  # User cancelled
            self.selected_points = []
            return
        
        # Use current length if l0 is empty, otherwise use provided value
        l0 = current_length if properties['l0'] == current_length else properties['l0']
        
        pulley_conn = [n1, n2, n3, properties['k'], 0, l0]
        self.pulley_matrix.append(pulley_conn)
        
        # Add visual lines
        line1 = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                            'r-', linewidth=3, alpha=0.8, label='Pulley')[0]
        line2 = self.ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], 
                            'r--', linewidth=3, alpha=0.8)[0]
        self.connection_lines.append(('pulley', line1))
        self.connection_lines.append(('pulley', line2))
        
        self.selected_points = []
        print(f"✓ Pulley created: nodes {n1}-{n2}-{n3}, k={properties['k']}, c={0}, l0={l0}")
        
    def undo_last(self):
        if not self.connection_lines:
            print("Nothing to undo.")
            return
            
        # Find the last connection type
        last_type = self.connection_lines[-1][0]
        
        if last_type == 'pulley':
            # Remove pulley (2 lines)
            if len(self.pulley_matrix) > 0:
                removed = self.pulley_matrix.pop()
                line1 = self.connection_lines.pop()[1]
                line2 = self.connection_lines.pop()[1]
                line1.remove()
                line2.remove()
                print(f"Undid pulley: {removed}")
        elif last_type == 'beam':
            # Remove beam (1 line)
            if len(self.beam_matrix) > 0:
                removed = self.beam_matrix.pop()
                line = self.connection_lines.pop()[1]
                line.remove()
                print(f"Undid beam: {removed}")
        elif last_type == 'spring':
            # Remove spring (1 line)
            if len(self.spring_matrix) > 0:
                removed = self.spring_matrix.pop()
                line = self.connection_lines.pop()[1]
                line.remove()
                print(f"Undid spring: {removed}")
            
    def reset_all(self):
        print("Resetting all connections...")
        
        # Clear all connections
        self.spring_matrix = []
        self.beam_matrix = []
        self.pulley_matrix = []
        
        # Remove all lines
        for conn_type, line in self.connection_lines:
            line.remove()
        self.connection_lines = []
        
        self.selected_points = []
        print("All connections reset.")
        
    def update_display(self):
        # Clear existing text labels
        for txt in self.ax.texts:
            txt.remove()
        
        if len(self.points) == 0:
            # No points to display
            if hasattr(self, 'point_scatter'):
                self.point_scatter.remove()
            self.point_scatter = self.ax.scatter([], [], [], 
                c='blue', s=80, picker=True, alpha=0.8, edgecolors='black')
        else:
            # Highlight selected points by changing their color
            colors = ['red' if i in self.selected_points else 'blue' for i in range(self.num_points)]
            sizes = [120 if i in self.selected_points else 80 for i in range(self.num_points)]
            
            # Remove old scatter and create new one
            if hasattr(self, 'point_scatter'):
                self.point_scatter.remove()
            self.point_scatter = self.ax.scatter(
                self.points[:, 0], self.points[:, 1], self.points[:, 2],
                c=colors, s=sizes, picker=True, alpha=0.8, edgecolors='black'
            )
            
            # Add point labels
            for i, point in enumerate(self.points):
                self.ax.text(point[0], point[1], point[2], f'{i}', fontsize=9, color='black')
        
        # Update instruction text
        self.info_text.set_text(self.get_instructions())
        self.update_title()
        
        # Keep axes equal only if we have points
        if len(self.points) > 0:
            self.set_equal_axes()
        
        self.fig.canvas.draw()
        
    def show_parameters(self):
        """Print current parameters"""
        print("\n" + "="*50)
        print("CURRENT PARAMETERS")
        print("="*50)
        print(f"Spring stiffness: {self.spring_k}")
        print(f"Spring damping: {self.spring_c}")
        print(f"Spring type: {self.spring_type}")
        print(f"Beam diameter: {self.beam_diameter}")
        print(f"Beam pressure: {self.beam_pressure}")
        print(f"Pulley stiffness: {self.pulley_k}")
        print(f"Pulley damping: {self.pulley_c}")
        print("="*50)
        print("To change parameters, edit the values in the code.")
            
    def show_help(self):
        """Show help in console"""
        help_text = """
Interactive Mesh Builder Help:

MOUSE:
- Left click: Select/deselect points (selected points turn red)
- In ADD POINT mode: Click anywhere to add a point (coordinates dialog will open)
- Mouse wheel: Zoom in/out
- Right click + drag: Pan view

KEYBOARD:
- 'a': Add point mode (click anywhere, enter coordinates in dialog)
- '1': Spring mode (select 2 points)
- '2': Beam mode (select 2 points) 
- '3': Pulley mode (select 3 points)
- '4': Select mode (select/deselect points without creating connections)
- 'c': Clear current selection
- 'u': Undo last connection
- 'r': Reset all connections
- 'e': Show current parameters
- 'h': Show this help
- 'q': Quit and print matrices

MODES:
- Add Point: Click anywhere and enter coordinates in dialog
- Select: Click points to select/deselect without creating connections
- Spring: Green lines, connects 2 points
- Beam: Blue thick lines, connects 2 points
- Pulley: Red lines, connects 3 points (n1->n2->n3)

WORKFLOW:
1. Start with empty mesh or loaded points
2. Press 'a' to add points, enter coordinates in dialog
3. Press '1', '2', or '3' to select connection mode
4. Press '4' for select mode to choose points without creating connections
5. Click on points to select them (they turn red)
6. Connection is created automatically when enough points are selected
7. Press 'q' when done to see the matrices

Point numbers are shown as text labels next to each point.
"""
        
        print(help_text)
        
    def quit_and_print(self):
        """Print all matrices and close"""
        print("\n" + "="*80)
        print("GENERATED MATRICES FOR KITE FEM")
        print("="*80)
        
        self.print_matrices()
        
        plt.close(self.fig)
        
    def run(self):
        """Start the interactive session"""
        print("Interactive Mesh Builder started!")
        print("="*50)
        print("INSTRUCTIONS:")
        print("1. Press 'a' to add points (enter coordinates in dialog)")
        print("2. Press '1' for spring mode, '2' for beam mode, '3' for pulley mode")
        print("3. Press '4' for select mode (select points without creating connections)")
        print("4. Click on points to select them (they turn red)")
        print("5. Connections are created automatically when enough points selected")
        print("6. Press 'h' for detailed help or 'q' to quit and print matrices")
        print("="*50)
        plt.show()


