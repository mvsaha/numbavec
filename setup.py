from distutils.core import setup

config = {
    'description': ('Bare bones dynamically sized vector numba.jitclass'
                    'for use in nopython mode.'),
    'author': 'Michael Saha',
    'url': 'https://github.com/mvsaha/numbavec',
    'download_url': 'https://github.com/mvsaha/numbavec',
    'version': '1.0.0',
    'packages': ['numbavec'],
    'name': 'numbavec',
    'requires': ['numpy', 'numba']
}

setup(**config)
