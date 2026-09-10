"""Transport integrity tests; no model or network dependencies."""
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import txt_transfer as tx


class TransferTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'output/setup/transfer_tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.tmp.name)
        self.source = self.root/'results'
        (self.source/'nested').mkdir(parents=True)
        (self.source/'RESULTS.txt').write_text('测试结果，只是传输测试，不是模型效果。\n')
        (self.source/'nested/过程.log').write_text('工具调用与日志😀，逐字保存。\n'*10000)
        (self.source/'empty.txt').write_bytes(b'')
        (self.source/'raw.log').write_bytes(bytes(range(256)))
        self.parts = self.root/'txt'
        tx.pack(self.source, self.parts)

    def tearDown(self):
        self.tmp.cleanup()

    def test_unicode_binary_and_empty_round_trip(self):
        report = tx.verify(self.parts, restored=self.root/'restored')
        self.assertGreater(report['parts'], 1)
        for path in self.parts.iterdir():
            self.assertLessEqual(path.stat().st_size, 90000)
            path.read_text(encoding='utf-8')
        for original in self.source.rglob('*'):
            if original.is_file():
                self.assertEqual(original.read_bytes(), (self.root/'restored'/original.relative_to(self.source)).read_bytes())
        tx.verify(self.parts, joined=self.root/'joined.txt')
        self.assertIn('测试结果，只是传输测试', (self.root/'joined.txt').read_text())

    def test_missing_part(self):
        next(self.parts.glob('part-*')).unlink()
        with self.assertRaises(ValueError): tx.verify(self.parts)

    def test_duplicate_part(self):
        shutil.copyfile(next(self.parts.glob('part-*')), self.parts/'duplicate.txt')
        with self.assertRaises(ValueError): tx.verify(self.parts)

    def test_tampered_or_truncated_part(self):
        path = next(self.parts.glob('part-*'))
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError): tx.verify(self.parts)

    def test_swapped_part_content(self):
        a, b = sorted(self.parts.glob('part-*'))[:2]
        raw_a, raw_b = a.read_bytes(), b.read_bytes()
        a.write_bytes(raw_b); b.write_bytes(raw_a)
        with self.assertRaises(ValueError): tx.verify(self.parts)

    def test_control_count_tamper(self):
        control = json.loads((self.parts/tx.CONTROL).read_text())
        control['parts'] += 1
        (self.parts/tx.CONTROL).write_text(json.dumps(control))
        with self.assertRaises(ValueError): tx.verify(self.parts)

    def test_unsafe_file_path(self):
        data = tx.MAGIC+tx.line({'path': '../escape.txt', 'bytes': 1, 'sha256': tx.digest(b'x'),
                                'encoding': 'utf-8', 'stored_bytes': 1})+b'x\n'+tx.line({'end':True,'files':1})
        with self.assertRaises(ValueError):
            tx.parse_stream(io.BytesIO(data), {'files':1,'stream_bytes':len(data)}, self.root/'unsafe')
        self.assertFalse((self.root/'escape.txt').exists())

    def test_no_overwrite(self):
        dest = self.root/'existing'
        dest.mkdir()
        (dest/'keep').write_text('keep')
        with self.assertRaises(ValueError): tx.verify(self.parts, restored=dest)
        self.assertEqual((dest/'keep').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main(verbosity=2)
