"""Decide what an uploaded file IS from its bytes, not its name.

Cloud Run logs, 2026-09-28 → 10-13: after the credits outage, every upload
error left in the log was a file the loader routed by extension and then could
not open. A legacy `.ppt` handed to python-pptx (it reads only `.pptx`), a file
named `.pptx` that was not a zip at all, a `.md` the file picker offers but the
router had no branch for. Each failure was deterministic, the student was told
"try uploading it again", and one retried three times in twenty seconds.

So the extension is a hint and the content decides. A failure that is about
the file raises `UploadFormatError` with a `code` the frontend turns into an
instruction she can act on ("save it as .pptx"), never "try again".
"""
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

# Codes the frontend knows (composerUploadModel.uploadFailureKind). Add one
# there before emitting it here, or she sees the generic message.
LEGACY_OFFICE = 'legacy_office'
NOT_A_DOCUMENT = 'not_a_document'
UNREADABLE_DOCUMENT = 'unreadable_document'
UNSUPPORTED_TYPE = 'unsupported_type'

TEXT_EXTENSIONS = ('.txt', '.md', '.markdown', '.csv')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff', '.heic', '.heif')
LEGACY_TARGET = {'.ppt': '.pptx', '.pps': '.pptx', '.doc': '.docx', '.xls': '.xlsx'}

OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
SOFFICE_TIMEOUT_S = 120


class UploadFormatError(ValueError):
    """The file itself can't be read. Retrying the same file won't help."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _head(path, size=1024):
    with open(path, 'rb') as handle:
        return handle.read(size)


def _image_kind(head):
    if head.startswith(b'\x89PNG'):
        return '.png'
    if head.startswith(b'\xff\xd8\xff'):
        return '.jpg'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return '.webp'
    if head[:6] in (b'GIF87a', b'GIF89a'):
        return '.gif'
    if head.startswith(b'BM'):
        return '.bmp'
    if head[:4] in (b'II*\x00', b'MM\x00*'):
        return '.tiff'
    if head[4:8] == b'ftyp' and head[8:12] in (b'heic', b'heix', b'mif1', b'msf1', b'heim', b'heis'):
        return '.heic'
    return None


def _zip_kind(path, ext):
    """An Office Open XML package says what it is in its part names."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return None
    if any(name.startswith('ppt/') for name in names):
        return '.pptx'
    if any(name.startswith('word/') for name in names):
        return '.docx'
    if any(name.startswith('xl/') for name in names):
        return '.xlsx'
    # A zip we can't place (Keynote, Pages, an archive of files).
    return ext if ext in ('.pptx', '.docx', '.xlsx') else 'zip'


def _looks_like_text(path):
    with open(path, 'rb') as handle:
        sample = handle.read(4096)
    if not sample or b'\x00' in sample:
        return False
    try:
        sample.decode('utf-8')
    except UnicodeDecodeError as error:
        # A multi-byte character cut by the 4096-byte window is still text.
        if error.start < len(sample) - 4:
            return False
    return True


def detect_format(path):
    """Return the extension the content really has, e.g. '.pptx'.

    Raises UploadFormatError when the content is a format we can't read.
    Legacy Office binaries return '.ppt'/'.doc'/'.xls' for `convert_legacy`.
    """
    ext = os.path.splitext(path)[-1].lower()
    head = _head(path)
    if not head:
        raise UploadFormatError(UNREADABLE_DOCUMENT, 'The file is empty.')
    # The spec lets a PDF header sit anywhere in the first 1024 bytes.
    if b'%PDF-' in head:
        return '.pdf'
    if head.startswith(b'PK\x03\x04'):
        kind = _zip_kind(path, ext)
        if kind == 'zip':
            raise UploadFormatError(UNSUPPORTED_TYPE, f"{ext or 'This'} file is a package we can't read.")
        if kind:
            return kind
        raise UploadFormatError(NOT_A_DOCUMENT, 'The file is damaged or not really a document.')
    if head.startswith(OLE_MAGIC):
        # The extension is the only clue to which legacy program wrote it.
        return ext if ext in LEGACY_TARGET else '.doc'
    image = _image_kind(head)
    if image:
        return image
    if ext in ('.docx', '.pptx', '.xlsx'):
        # Named like Office, but not the zip every Office file is.
        raise UploadFormatError(NOT_A_DOCUMENT, f'This {ext} file is damaged or was renamed from another format.')
    if _looks_like_text(path):
        return '.csv' if ext == '.csv' else '.txt'
    raise UploadFormatError(UNSUPPORTED_TYPE, f"We can't read {ext or 'this kind of'} files.")


# Where the installers put it when it is not on PATH (the Windows and macOS
# installers never add it), so a local dev backend converts like the image.
SOFFICE_CANDIDATES = (
    r'C:\Program Files\LibreOffice\program\soffice.exe',
    r'C:\Program Files (x86)\LibreOffice\program\soffice.exe',
    '/Applications/LibreOffice.app/Contents/MacOS/soffice',
)


def soffice_path():
    found = shutil.which('soffice') or shutil.which('libreoffice')
    if found:
        return found
    return next((path for path in SOFFICE_CANDIDATES if os.path.exists(path)), None)


def convert_legacy(path, legacy_ext, out_dir):
    """Convert a legacy Office binary with LibreOffice; return the new path.

    Without LibreOffice on the machine this raises the instruction instead:
    she can do the conversion herself in ten seconds.
    """
    target = LEGACY_TARGET[legacy_ext]
    binary = soffice_path()
    if not binary:
        raise UploadFormatError(
            LEGACY_OFFICE,
            f'This is an older {legacy_ext} file. Save it as {target} or PDF and upload that.')
    # LibreOffice names the output after the input, so give it a clean name.
    source = os.path.join(out_dir, 'source' + legacy_ext)
    shutil.copyfile(path, source)
    # A private profile per call: two conversions sharing the default profile
    # lock each other out.
    profile = tempfile.mkdtemp(dir=out_dir)
    try:
        subprocess.run(
            # as_uri: file:///C:/… on Windows, file:///tmp/… on Linux.
            [binary, f'-env:UserInstallation={Path(profile).as_uri()}',
             '--headless', '--convert-to', target.lstrip('.'), '--outdir', out_dir, source],
            check=True, timeout=SOFFICE_TIMEOUT_S, capture_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise UploadFormatError(
            LEGACY_OFFICE,
            f'This older {legacy_ext} file could not be converted. Save it as {target} or PDF and upload that.'
        ) from error
    converted = os.path.join(out_dir, 'source' + target)
    if not os.path.exists(converted):
        raise UploadFormatError(
            LEGACY_OFFICE,
            f'This older {legacy_ext} file could not be converted. Save it as {target} or PDF and upload that.')
    return converted
