from ipywidgets import HTML


DEFAULT_COLOR = 'white'
DEFAULT_SIZE = '20px'


class _Spinner(HTML):
    """
    Baseclass of HTML implemented spinner.

    Parameters
    ----------
    display : bool, optional
        Spinner will be spinning on creation if True.
    color : str, optional
        Spinner color.
    size : str, optional
        Typically refers to the font size of the text.
    layout_dict : dict, optional
        Dictionary of layout paramaters to pass to HTML widget.

    Methods
    -------
    start
        Start spinning.
    stop
        Stop spinning.
    toggle
        Switch widget state, e.g. stop of already spinning or vice versa.
    """

    def __init__(
        self,
        display=False,
        color=DEFAULT_COLOR,
        size=DEFAULT_SIZE,
        text='Loading...',
        layout_dict=None,
        *args,
        **kwargs
    ):

        self._widget_generated = False

        self.color = color
        self.size = size
        self.layout_dict = layout_dict
        self.text = text

        body = self._gen_html_body()

        super().__init__(body, layout=self.layout_dict)

        self._widget_generated = True

        if display:
            self.start()
        else:
            self.stop()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self._update_value()

    @property
    def layout_dict(self):
        return self._layout_dict

    @layout_dict.setter
    def layout_dict(self, value):
        if value is None:
            value = {}
        self._layout_dict = value
        self._update_value()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
        self._update_value()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        self._update_value()

    def _update_value(self):
        """
        Update HTML value.
        """
        # This method is used inside properties so we should only update the
        # HTML value if the widget has already been created.
        if self._widget_generated:
            self.value = self._gen_html_body()

    def _gen_html_body(self):
        """
        Return HTML body.
        """
        pass

    def start(self):
        """
        Start the widget spinning.
        """
        self.layout.display = ''

    def stop(self):
        """
        Stop the widget spinning.
        """
        self.layout.display = 'None'

    def toggle(self):
        """
        Toggle the state of the widget, i.e. if it's spinning then stop it and
        vice versa.
        """
        if self.layout.display == '':
            self.stop()
        elif self.layout.display == 'None':
            self.start()


class Spinner(_Spinner):
    """
    Basic spinner with optional text.
    """

    def __init__(self, *args, **kwargs):
        self.spinner_type = 'fa fa-spinner fa-spin'
        super().__init__(*args, **kwargs)

    def _gen_html_body(self):
        icon = f'<i class="{self.spinner_type}"></i>'
        body = icon + '&nbsp;&nbsp;' + self.text
        return (
            f'<p style="color:{self.color}; font-size:{self.size};">{body}</p>'
        )
