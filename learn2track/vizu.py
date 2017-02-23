#! /usr/bin/env python
import collections

import os
import numpy as np
import textwrap
from os.path import join as pjoin

import nibabel as nib
from nibabel.streamlines import Tractogram

from dipy.data import fetch_viz_icons, read_viz_icons
from dipy.fixes import argparse

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric as MDF

from dipy.viz import window, actor, gui_2d, utils, gui_follower
from dipy.viz.colormap import distinguishable_colormap, line_colors

from dipy.viz.interactor import CustomInteractorStyle


metric = MDF(ResampleFeature(nb_points=30))
darkcolors = [(0.1, 0, 0), (0.1, 0.1, 0), (0.1, 0.1, 0.1),
              (0, 0.1, 0), (0, 0.1, 0.1),
              (0, 0, 0.1), (0.1, 0, 0.1)]


def animate_button_callback(iren, obj, button):
    """ General purpose callback that dims a button. """
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # button: Button2D
    color = np.asarray(obj.GetProperty().GetColor())
    obj.GetProperty().SetColor(*(color*0.5))
    iren.force_render()
    iren.event.abort()  # Stop propagating the event.


class Bundle(object):
    def __init__(self, streamlines, centroid=None, threshold_used=np.inf, color=None):
        self.streamlines = streamlines
        self.color = color
        self.centroid = centroid
        self.clusters = []
        self.clusters_colors = [None]
        self.streamlines_colors = np.ones((len(self.streamlines), 3))
        self.threshold_used = threshold_used
        self.last_threshold = None
        self.ren = None

        # Create 3D actor to display this bundle's streamlines.
        self.actor = actor.line(self.streamlines, colors=color)
        self.centroid_actors = []
        self.has_changed = True
        self.centroids_visible = False
        self.streamlines_visible = True

        #
        lines_range = range(len(self.streamlines))
        points_per_line = self.streamlines._lengths.astype(np.intp)
        cols_arr = line_colors(self.streamlines)
        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
        self.original_colors = cols_arr[colors_mapper]
        if self.color is not None:
            self.original_colors[:] = self.color

        self._cluster(np.inf)

    def add_to_renderer(self, ren):
        self.ren = ren
        ren.add(self.actor)

    def remove_from_renderer(self, ren):
        ren.rm(self.actor)
        ren.rm(*self.centroid_actors)
        self.centroid_actors = []

    def update(self):
        if not self.has_changed:
            return  # Nothing changed

        self.has_changed = False

        # Update centroids actor.
        # Remove old centroid actors.
        self.ren.rm(*self.centroid_actors)

        # Create an actor for every centroid.
        self.centroid_actors = []
        for cluster, color in zip(self.clusters, self.clusters_colors):
            centroid_actor = actor.streamtube([cluster.centroid], colors=color,
                                              linewidth=0.1+0.1*np.log(len(cluster)))
            centroid_actor.SetVisibility(self.centroids_visible)
            self.ren.add(centroid_actor)
            # TODO add a click callback
            self.centroid_actors.append(centroid_actor)

    def show_centroids(self):
        self.update()
        self.centroids_visible = True
        for a in self.centroid_actors:
            a.SetVisibility(True)

    def hide_centroids(self):
        self.centroids_visible = False
        for a in self.centroid_actors:
            a.SetVisibility(False)

    def show_streamlines(self):
        self.streamlines_visible = True
        self.actor.SetVisibility(True)

    def hide_streamlines(self):
        self.streamlines_visible = False
        self.actor.SetVisibility(False)

    def preview(self, threshold):
        self._cluster(threshold)
        self.update()
        return len(self.clusters)

    def reset(self):
        self._cluster(np.inf)
        self.update()

    def _cluster(self, threshold):
        qb = QuickBundles(metric=metric, threshold=threshold)
        self.last_threshold = threshold

        self.clusters = qb.cluster(self.streamlines)
        self.clusters_colors = [color for c, color in zip(self.clusters, distinguishable_colormap(bg=(0, 0, 0), exclude=darkcolors))]

        if threshold < np.inf:
            print("{} clusters with QB threshold of {}mm".format(len(self.clusters), threshold))

        # Keep a flag telling there have been changes.
        self.has_changed = True

        if len(self.clusters) == 1:
            # Keep initial color
            self.clusters_colors = [self.color] if self.color is not None else [(0, 0, 1)]
            vtk_colors = utils.numpy_to_vtk_colors(255 * self.original_colors)
            vtk_colors.SetName("Colors")
            self.actor.GetMapper().GetInput().GetPointData().SetScalars(vtk_colors)
            return

        for cluster, color in zip(self.clusters, self.clusters_colors):
            self.streamlines_colors[cluster.indices] = color

        self.colors = []
        for color, streamline in zip(self.streamlines_colors, self.streamlines):
            self.colors += [color] * len(streamline)

        vtk_colors = utils.numpy_to_vtk_colors(255 * np.array(self.colors))
        vtk_colors.SetName("Colors")
        self.actor.GetMapper().GetInput().GetPointData().SetScalars(vtk_colors)

    def get_cluster_as_bundles(self):
        if self.clusters is None:
            raise NameError("Streamlines need to be clustered first!")

        if len(self.clusters) == 1:
            # Handle this case separatly just so we can keep the initial bundle color.
            bundle = Bundle(self.streamlines[self.clusters[0].indices], self.clusters[0].centroid, self.last_threshold, self.color)
            return [bundle]

        bundles = []
        for cluster, color in zip(self.clusters, self.clusters_colors):
            bundle = Bundle(self.streamlines[cluster.indices], cluster.centroid, self.last_threshold, color)
            bundles.append(bundle)

        return bundles


class StreamlinesVizu(object):
    def __init__(self, tractogram, anat=None, screen_size=(1360, 768), default_clustering_threshold=None, verbose=False):
        self.screen_size = screen_size
        self.default_clustering_threshold = default_clustering_threshold
        self.verbose = verbose

        self.cpt = None  # Used for iterating through the clusters.
        self.bundles = {}
        self.root_bundle = "/"
        self.keys = [self.root_bundle]
        self.bundles[self.root_bundle] = Bundle(tractogram.streamlines)
        self.selected_bundle = None
        self.last_threshold = None
        self.last_bundles_visibility_state = "dimmed"
        self.anat = anat
        self.anat_actor = None

    def _set_bundles_visibility(self, state, bundles=None, exclude=[]):
        if bundles is None:
            bundles = list(self.bundles.values())

        if state == "visible":
            self.show_dim_hide_button.color = (0, 1, 0)
            # self.last_bundles_visibility_state = "visible"
            visibility = True
            opacity = 1

        elif state == "dimmed":
            self.show_dim_hide_button.color = (0, 0, 1)
            self.last_bundles_visibility_state = "dimmed"
            visibility = True
            opacity = 0.6

        elif state == "hidden":
            self.show_dim_hide_button.color = (1, 0, 0)
            self.last_bundles_visibility_state = "hidden"
            visibility = False
            opacity = 1

        else:
            raise ValueError("Unknown visibility state: {}".format(state))

        # Make the changes
        for bundle in bundles:
            if bundle not in exclude:
                bundle.actor.SetVisibility(visibility)
                if opacity < 1:
                    opacity = max(0.1, opacity - 0.1 * np.log10(len(bundle.streamlines)))
                bundle.actor.GetProperty().SetOpacity(opacity)

    def add_bundle(self, bundle_name, bundle):
        self.keys.append(bundle_name)
        self.keys = sorted(self.keys)
        self.ren.add(bundle)
        self.bundles[bundle_name] = bundle

    def remove_bundle(self, bundle_name):
        self.keys.remove(bundle_name)
        self.keys = sorted(self.keys)
        self.ren.rm(self.bundles[bundle_name])
        del self.bundles[bundle_name]

    def select_next(self):
        # Sort bundle according to their bundle size.
        indices = np.lexsort((self.keys, [len(self.bundles[k].streamlines) for k in self.keys])).tolist()[::-1]

        if self.selected_bundle is None:
            cpt = 0
        else:
            cpt = indices.index(self.keys.index(self.selected_bundle))
            cpt = (cpt + 1) % len(self.keys)

        self.select(self.keys[indices[cpt]])
        print("({}/{})".format(cpt+1, len(self.keys)))

    def select_previous(self):
        # Sort bundle according to their bundle size.
        indices = np.lexsort((self.keys, [len(self.bundles[k].streamlines) for k in self.keys])).tolist()[::-1]

        if self.selected_bundle is None:
            cpt = 0
        else:
            cpt = indices.index(self.keys.index(self.selected_bundle))
            cpt = (cpt - 1) % len(self.keys)

        self.select(self.keys[indices[cpt]])
        print("({}/{})".format(cpt+1, len(self.keys)))

    def select(self, bundle_name=None):
        # Unselect first, if possible.
        if self.selected_bundle is not None and self.selected_bundle in self.bundles:
            bundle = self.bundles[self.selected_bundle]
            bundle.reset()

        if bundle_name is None:
            # Close panels
            self.selected_bundle = None
            self.clustering_panel.set_visibility(False)
            self._set_bundles_visibility("visible")
            self.iren.force_render()
            self.cpt = None  # Used for iterating through the clusters.
            return

        self.selected_bundle = bundle_name
        bundle = self.bundles[self.selected_bundle]
        print("Selecting {} streamlines...".format(len(bundle.streamlines)))

        # Set maximum threshold value depending on the selected bundle.
        self.clustering_panel.slider.max_value = bundle.actor.GetLength() / 2.
        if self.default_clustering_threshold is None:
            self.clustering_panel.slider.set_ratio(1)
        else:
            self.clustering_panel.slider.set_value(self.default_clustering_threshold)
        self.clustering_panel.slider.update()
        self.clustering_panel.set_visibility(True)

        # Dim other bundles
        self._set_bundles_visibility("visible", bundles=[bundle])
        self._set_bundles_visibility(self.last_bundles_visibility_state, exclude=[bundle])
        bundle.preview(threshold=self.clustering_panel.slider.value)

        self.iren.force_render()

    def _add_bundle_right_click_callback(self, bundle, bundle_name):

        def open_clustering_panel(iren, obj, *args):
            self.select(bundle_name)
            iren.event.abort()  # Stop propagating the event.

        def ctrl_leftcklick_open_clustering_panel(iren, obj, *args):
            if not iren.event.ctrl_key:
                return

            self.select(bundle_name)
            iren.event.abort()  # Stop propagating the event.

        self.iren.add_callback(bundle.actor, "RightButtonPressEvent", open_clustering_panel)
        self.iren.add_callback(bundle.actor, "LeftButtonPressEvent", ctrl_leftcklick_open_clustering_panel)  # Support for MAC OSX

    def _make_clustering_panel(self):
        # Panel
        size = (self.screen_size[0], self.screen_size[1]//10)
        center = tuple(np.array(size) / 2.)  # Lower left corner of the screen.
        panel = gui_2d.Panel2D(center=center, size=size, color=(1, 1, 1), align="left")

        # Nb. clusters label
        label = gui_2d.Text2D("# clusters")
        panel.add_element(label, (0.01, 0.2))

        # "Apply" button
        def apply_button_callback(iren, obj, button):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D
            bundles = self.bundles[self.selected_bundle].get_cluster_as_bundles()
            print("Preparing the new {} clusters...".format(len(bundles)))

            # Create new actors, one for each new bundle.
            # Sort bundle in decreasing size.
            for i, bundle in enumerate(bundles):
                bundle_name = "{}{}/".format(self.selected_bundle, i)
                self.add_bundle(bundle_name, bundle)
                self._add_bundle_right_click_callback(bundle, bundle_name)

            # Remove original bundle.
            self.remove_bundle(self.selected_bundle)
            self.select(None)

            # TODO: apply clustering if needed, close panel, add command to history, re-enable bundles context-menu.
            button.color = (0, 1, 0)  # Restore color.
            print("Done.")
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        button = gui_2d.Button2D(icon_fnames={'apply': read_viz_icons(fname='checkmark_neg.png')})
        button.color = (0, 1, 0)
        button.add_callback("LeftButtonPressEvent", animate_button_callback)
        button.add_callback("LeftButtonReleaseEvent", apply_button_callback)
        panel.add_element(button, (0.98, 0.2))

        # "Hide" button
        def toggle_other_bundles_visibility(iren, *args):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D

            if self.last_bundles_visibility_state == "dimmed":
                self.last_bundles_visibility_state = "hidden"
                self._set_bundles_visibility("hidden", exclude=[self.bundles[self.selected_bundle]])

            elif self.last_bundles_visibility_state == "hidden":
                self.last_bundles_visibility_state = "dimmed"
                self._set_bundles_visibility("dimmed", exclude=[self.bundles[self.selected_bundle]])

            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        self.show_dim_hide_button = gui_2d.Button2D(icon_fnames={'show_dim_hide': read_viz_icons(fname='infinite_neg.png')})
        self.show_dim_hide_button.add_callback("LeftButtonPressEvent", toggle_other_bundles_visibility)
        panel.add_element(self.show_dim_hide_button, (0.02, 0.88))

        # Threshold slider
        def disk_press_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            # Only need to grab the focus.
            iren.event.abort()  # Stop propagating the event.

        def disk_move_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D

            # Reset textbox
            textbox = slider.textbox.actor
            if textbox in iren.active_props:
                iren.remove_active_prop(textbox)

            position = iren.event.position
            slider.set_position(position)

            threshold = slider.value
            if self.last_threshold != threshold:
                nb_bundles = self.bundles[self.selected_bundle].preview(threshold)
                self.last_threshold = threshold
                label.set_message("{} clusters".format(nb_bundles))

            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        # Slider textbox
        def slider_textbox_select_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            iren.add_active_prop(slider.textbox.actor)
            slider.textbox.set_message("")
            slider.textbox.caret_pos = 0
            slider.textbox.render_text(show_caret=True)
            iren.force_render()

        def slider_textbox_keypress_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D

            key = iren.event.key.lower()
            textbox = slider.textbox

            if key not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "period", "backspace", "return", "kp_enter"]:
                # Unauthorized key
                pass
            elif len(textbox.text) == 0 and key in ["return", "kp_enter"]:
                # User pressed "enter" on empty field; reset textbox
                iren.remove_active_prop(textbox.actor)
                textbox.set_message(slider.format_text())
            elif len(textbox.text) >= 4 and key not in ["backspace", "return", "kp_enter"]:
                # Textbox is filled to max length
                pass
            else:
                # Switch for period character
                key = '.' if key == "period" else key

                # Process keypress
                is_done = textbox.handle_character(key)
                if is_done:
                    iren.remove_active_prop(textbox.actor)

                    try:
                        threshold = float(textbox.text)

                        if threshold > slider.max_value:
                            # Invalid value, reset textbox
                            textbox.set_message(slider.format_text())
                        elif self.last_threshold != threshold:
                            slider.set_value(threshold)

                            nb_bundles = self.bundles[self.selected_bundle].preview(threshold)
                            self.last_threshold = threshold
                            label.set_message("{} clusters".format(nb_bundles))

                    except ValueError:
                        # Invalid value, reset textbox
                        textbox.set_message(slider.format_text())

            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        slider = gui_2d.LineSlider2D(length=1000, text_template="{value:.1f}mm")
        slider.add_callback("LeftButtonPressEvent", disk_move_callback, slider.slider_line)
        slider.add_callback("LeftButtonPressEvent", disk_press_callback, slider.slider_disk)
        slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_disk)
        slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_line)

        slider.add_callback("KeyPressEvent", slider_textbox_keypress_callback, slider.text)
        slider.add_callback("LeftButtonPressEvent", slider_textbox_select_callback, slider.text)

        panel.add_element(slider, (0.5, 0.5))
        panel.slider = slider

        # Add shortcut keys.
        def toggle_visibility_onchar_callback(iren, evt_name):
            if self.selected_bundle is None:
                return

            if iren.event.key.lower() == "space":
                toggle_other_bundles_visibility(iren)

        self.iren.AddObserver("CharEvent", toggle_visibility_onchar_callback)

        return panel

    def _make_anatomy_panel(self, axial_slicer, sagittal_slicer, coronal_slicer):
        # Panel
        size = (self.screen_size[0]//8, self.screen_size[1]//6)
        center = (size[0] / 2., np.ceil(self.screen_size[1] / 10. + size[1]) )  # Lower left corner of the screen.
        panel = gui_2d.Panel2D(center=center, size=size, color=(0., 0., 0.), align="left")

        # Create all sliders that will be responsible of moving the slices of the anatomy.
        length = size[0] - 10
        text_template = lambda obj: "{value:}".format(value=int(obj.value))
        axial_slider = gui_2d.LineSlider2D(length=length, text_template=text_template)
        coronal_slider = gui_2d.LineSlider2D(length=length, text_template=text_template)
        sagittal_slider = gui_2d.LineSlider2D(length=length, text_template=text_template)

        # Common to all sliders.
        def disk_press_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            # Only need to grab the focus.
            iren.event.abort()  # Stop propagating the event.

        def axial_disk_move_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            position = iren.event.position
            slider.set_position(position)
            # Move slices accordingly.
            axial_slicer.display(x=int(slider.value))
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        def coronal_disk_move_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            position = iren.event.position
            slider.set_position(position)
            # Move slices accordingly.
            coronal_slicer.display(y=int(slider.value))
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        def sagittal_disk_move_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            position = iren.event.position
            slider.set_position(position)
            # Move slices accordingly.
            sagittal_slicer.display(z=int(slider.value))
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        # Add callbacks to the sliders.
        axial_slider.add_callback("LeftButtonPressEvent", axial_disk_move_callback, axial_slider.slider_line)
        axial_slider.add_callback("LeftButtonPressEvent", disk_press_callback, axial_slider.slider_disk)
        axial_slider.add_callback("MouseMoveEvent", axial_disk_move_callback, axial_slider.slider_disk)
        axial_slider.add_callback("MouseMoveEvent", axial_disk_move_callback, axial_slider.slider_line)
        axial_slider.max_value = axial_slicer.shape[0]
        axial_slider.set_ratio(0.5)
        axial_slider.update()

        coronal_slider.add_callback("LeftButtonPressEvent", coronal_disk_move_callback, coronal_slider.slider_line)
        coronal_slider.add_callback("LeftButtonPressEvent", disk_press_callback, coronal_slider.slider_disk)
        coronal_slider.add_callback("MouseMoveEvent", coronal_disk_move_callback, coronal_slider.slider_disk)
        coronal_slider.add_callback("MouseMoveEvent", coronal_disk_move_callback, coronal_slider.slider_line)
        coronal_slider.max_value = coronal_slicer.shape[1]
        coronal_slider.set_ratio(0.5)
        coronal_slider.update()

        sagittal_slider.add_callback("LeftButtonPressEvent", sagittal_disk_move_callback, sagittal_slider.slider_line)
        sagittal_slider.add_callback("LeftButtonPressEvent", disk_press_callback, sagittal_slider.slider_disk)
        sagittal_slider.add_callback("MouseMoveEvent", sagittal_disk_move_callback, sagittal_slider.slider_disk)
        sagittal_slider.add_callback("MouseMoveEvent", sagittal_disk_move_callback, sagittal_slider.slider_line)
        sagittal_slider.max_value = sagittal_slicer.shape[2]
        sagittal_slider.set_ratio(0.5)
        sagittal_slider.update()

        # Add the slicers to the panel.
        panel.add_element(axial_slider, (0.5, 0.15))
        panel.add_element(coronal_slider, (0.5, 0.5))
        panel.add_element(sagittal_slider, (0.5, 0.85))

        # Initialize slices of the anatomy.
        axial_slicer.display(x=int(axial_slider.value))
        coronal_slicer.display(y=int(coronal_slider.value))
        sagittal_slicer.display(z=int(sagittal_slider.value))

        return panel

    def initialize_scene(self):
        self.ren = window.Renderer()
        self.iren = CustomInteractorStyle()
        self.show_m = window.ShowManager(self.ren, size=self.screen_size, interactor_style=self.iren)

        # Add clustering panel to the scene.
        self.clustering_panel = self._make_clustering_panel()
        self.clustering_panel.set_visibility(False)
        self.ren.add(self.clustering_panel)

        # Add "Reset/Home" button
        def reset_button_callback(iren, obj, button):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D
            print("Merging remaining bundles...")

            streamlines = nib.streamlines.ArraySequence()
            for k, bundle in self.bundles.items():
                streamlines.extend(bundle.streamlines)
                self.remove_bundle(k)

            if len(streamlines) == 0:
                print("No streamlines left to merge.")
                iren.force_render()
                iren.event.abort()  # Stop propagating the event.
                return

            # Add new root bundle to the scene.
            self.add_bundle(self.root_bundle, Bundle(streamlines))
            self._add_bundle_right_click_callback(self.bundles[self.root_bundle], self.root_bundle)
            self.select(None)

            print("{} streamlines merged.".format(len(streamlines)))
            button.color = (1, 1, 1)  # Restore color.
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        reset_button = gui_2d.Button2D(icon_fnames={'reset': read_viz_icons(fname='home3_neg.png')})
        reset_button.color = (1, 1, 1)
        reset_button.add_callback("LeftButtonPressEvent", animate_button_callback)
        reset_button.add_callback("LeftButtonReleaseEvent", reset_button_callback)
        reset_button.set_center((self.screen_size[0] - 20, self.screen_size[1] - 60))
        self.ren.add(reset_button)


        # Add toggle "Centroid/Streamlines" button
        def centroids_toggle_button_callback(iren, obj, button):
            if button.current_icon_name == "streamlines":
                button.next_icon()
                for bundle in self.bundles.values():
                    bundle.show_centroids()
                    bundle.hide_streamlines()

            elif button.current_icon_name == "centroids":
                button.next_icon()
                for bundle in self.bundles.values():
                    bundle.hide_centroids()
                    bundle.show_streamlines()

            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        centroids_toggle_button = gui_2d.Button2D(icon_fnames={'streamlines': read_viz_icons(fname='database_neg.png'),
                                                               'centroids': read_viz_icons(fname='centroid_neg.png')})
        centroids_toggle_button.color = (1, 1, 1)
        # centroids_toggle_button.add_callback("LeftButtonPressEvent", animate_button_callback)
        centroids_toggle_button.add_callback("LeftButtonReleaseEvent", centroids_toggle_button_callback)
        centroids_toggle_button.set_center((20, self.screen_size[1] - 20))
        self.ren.add(centroids_toggle_button)

        # Add objects to the scene.
        self.ren.add(self.bundles[self.root_bundle])
        self._add_bundle_right_click_callback(self.bundles[self.root_bundle], self.root_bundle)

        # Add shortcut keys.
        def select_biggest_cluster_onchar_callback(iren, evt_name):
            if self.verbose:
                print("Pressed {} (shift={}), (ctrl={}), (alt={})".format(
                    iren.event.key, iren.event.shift_key, iren.event.ctrl_key, iren.event.alt_key))

            if iren.event.key.lower() == "escape":
                self.select(None)

            elif "tab" in iren.event.key.lower():
                if iren.event.shift_key:
                    self.select_previous()
                else:
                    self.select_next()
            elif iren.event.key == "c":
                for bundle in self.bundles.values():
                    bundle.show_centroids()
                    bundle.hide_streamlines()
                iren.force_render()
            elif iren.event.key == "C":
                for bundle in self.bundles.values():
                    bundle.hide_centroids()
                    bundle.show_streamlines()
                iren.force_render()

            iren.event.abort()  # Stop propagating the event.

        self.iren.AddObserver("CharEvent", select_biggest_cluster_onchar_callback)

        # Add anatomy, if there is one.
        if self.anat is not None:
            anat_data = self.anat.get_data()
            if anat_data.ndim == 4:
                # Take b0 (assuming it is diffusion data)
                anat_data = anat_data[..., 0]

            self.anat_axial_slicer = actor.slicer(anat_data, affine=self.anat.affine)
            self.anat_coronal_slicer = actor.slicer(anat_data, affine=self.anat.affine)
            self.anat_sagittal_slicer = actor.slicer(anat_data, affine=self.anat.affine)
            self.ren.add(self.anat_axial_slicer, self.anat_coronal_slicer, self.anat_sagittal_slicer)
            self.anatomy_panel = self._make_anatomy_panel(self.anat_axial_slicer, self.anat_coronal_slicer, self.anat_sagittal_slicer)
            self.ren.add(self.anatomy_panel)

    def run(self):
        self.show_m.start()


def check_dataset_integrity(dataset, subset=1):
    assert len(dataset.subjects) == 1, "Only support dataset with only one subject for now."

    fetch_viz_icons()
    tractogram = nib.streamlines.Tractogram(dataset.streamlines)

    if subset < 1:
        rng = np.random.RandomState(1234)
        idx = np.arange(len(tractogram))
        rng.shuffle(idx)
        tractogram = tractogram[idx[:int(subset*len(tractogram))]]

    anat = dataset.subjects[0].signal

    # In a `TractographyDataset` object, streamlines are supposed to be in voxel space.
    # We will bring the streamline into rasmm as they should be displayed.
    tractogram.apply_affine(anat.affine)

    vizu = StreamlinesVizu(tractogram, anat=anat, screen_size=(800, 600))
    vizu.initialize_scene()
    vizu.run()
