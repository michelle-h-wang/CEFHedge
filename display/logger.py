"""
This module contains custom widgets.
"""

from collections import deque
import datetime as dt

from bqwidgets import TickerAutoComplete
from IPython.display import display
from ipywidgets import HTML


LAYOUT_DICT = {
    'display': 'flex',
    'max_height': '75px',
    'max_width': '600px',
    'overflow_y': 'auto',
    'margin': '20px 0px 20px 0px',
    'border': '1px solid #505050',
    'padding': '8px 5px 8px 8px'
}


class ApplicationLogger(object):
    """GUI for an HTML widget that can provide information to the user through
    what is effectively a console window.

    Args:
        - max_msgs (int): Maximum number of messages to store in logger.
        - layout_dict (dict or None): Dictionary of layout params.

    Attributes:
        - widgets (dict of widgets): Store all the associated widgets.
        - msg_queue (deque): Stores logger messages.

    Examples:
        >>> logger = ApplicationLogger()
        >>> logger.log_message('Hello world', color='green')
    """

    def __init__(self, max_msgs=20, layout_dict=None, **kwargs):

        if layout_dict is None:
            layout_dict = LAYOUT_DICT

        self._kwargs = kwargs

        self.widgets = dict()
        self.msg_queue = deque(maxlen=max_msgs)

        validated_layout_dict = self.__validate_layout_dict(layout_dict)
        self.__create_html_widget(validated_layout_dict)

    def __validate_layout_dict(self, layout_dict):
        """Checks the layout_dict provided when the object is instantiated to
        see if it is None. If it is None then we utilize the default CSS layout
        dictionary in this class.

        Args:
            - layout_dict (dict or None): CSS property dictionary provided at
                                          object instantiation

        Returns:
            - validated_layout_dict (dict): CSS property dictionary for widget
        """
        if layout_dict is None:
            validated_layout_dict = self.__get_default_layout_dict()
        else:
            validated_layout_dict = layout_dict
        return validated_layout_dict

    def __get_default_layout_dict(self):
        """Return a default layout dict for the styling of the logger.

        Returns:
            - default_layout (dict): dictionary containing default
                                     CSS layout parameters
        """
        default_layout = {'display': 'flex',
                          'max_height': '75px',
                          'max_width': '600px',
                          'overflow_y': 'auto',
                          'margin': '0px 0px 0px 0px',
                          'border': '1px solid grey'}
        return default_layout

    def log_message(self, msg, color=None):
        """Add a message and update the widget.

        Args:
            - msg (str): Message to display in HTML console.
            - color (optional, str): HTML color of the text.
        """

        if color is not None:
            template = '<font color="{font_color}">{user_msg}</font>'
            msg = template.format(font_color=str(color), user_msg=msg)

        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        modified_msg = "%s - %s" % (timestamp, msg)
        self.msg_queue.appendleft(modified_msg)

        self.__update_html_console()

    def __create_html_widget(self, validated_layout_dict):
        """
        Summary:
            Creates the HTML widget which will be used
            to display our messages and store it in
            the self.widgets dictionary
        Args:
            - validated_layout_dict (dict): dictionary containing CSS
                                            properties for widget layout
        """
        widget = HTML('', layout=validated_layout_dict, **self._kwargs)
        self.widgets['html_console'] = widget

    def __update_html_console(self):
        """
        Summary:
            Updates the HTML widget console with the
            latest set of messages
        """
        # concatenate the message strings into HTML format
        html_string = "<br>".join(list(self.msg_queue))
        # update the 'value' of the HTML widget to display
        # the latest messages to the user
        self.widgets['html_console'].value = html_string

    def get_widget(self):
        """
        Summary:
            This function is called by an external application
            to obtain the underlying widget for when we
            display the external application to the user.
        Returns:
            - widget_html_console (ipywidgets.HTML):
                HTML widget with our displayed messages
        """
        widget_html_console = self.widgets['html_console']
        return widget_html_console

    def display_widget(self):
        """
        Summary:
            This function forces Jupyter Notebook to display
            our widget. This should not be used in most instances
            since this widget is part of a larger application.
        """
        display(self.widgets['html_console'])
