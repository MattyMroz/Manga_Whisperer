# magi.py
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import List
import re

from PIL import Image
import numpy as np
import torch
from transformers import AutoModel
import typer
from rich import print
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TimeRemainingColumn


class MangaTranscriber:
    input: Path = None
    transcription: Path = None
    transcription_images: Path = None
    skip_existing: bool = None
    log: logging.Logger = None
    supported_extensions: List[str] = [
        "bmp",
        "dib",
        "jpeg",
        "jpg",
        "jpe",
        "jp2",
        "png",
        "webp",
        "pbm",
        "pgm",
        "ppm",
        "pxm",
        "pnm",
        "pfm",
        "sr",
        "ras",
        "tiff",
        "tif",
        "exr",
        "hdr",
        "pic",
        "gif",
        "tga",
    ]

    def __init__(
        self,
        input: Path,
        transcription: Path,
        transcription_images: Path,
        skip_existing: bool = False,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        self.input = input.resolve()
        self.transcription = transcription.resolve()
        self.transcription_images = transcription_images.resolve()
        self.skip_existing = skip_existing
        self.log = log
        self.log.info("Ładuję model MAGI...")
        try:
            self.model = AutoModel.from_pretrained(
                "ragavsachdeva/magi", trust_remote_code=True)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.log.info("Model MAGI załadowany na GPU.")
            else:
                self.log.warning("GPU niedostępne. Model MAGI działa na CPU.")
        except Exception as e:
            self.log.error(f"Błąd podczas ładowania modelu MAGI: {str(e)}")
            raise

    def run(self) -> None:
        self.log.info(f"Rozpoczynam przetwarzanie w katalogu: {self.input}")
        if not self.input.exists():
            self.log.error(f'Folder "{self.input}" nie istnieje.')
            sys.exit(1)
        elif self.input.is_file():
            self.log.error(f'"{self.input}" jest plikiem, a nie folderem.')
            sys.exit(1)

        self.transcription.mkdir(parents=True, exist_ok=True)
        self.transcription_images.mkdir(parents=True, exist_ok=True)

        images = list(self.input.rglob("*.*"))
        images = [img for img in images if img.suffix.lower()[1:]
                  in self.supported_extensions]

        self.log.info(f"Znaleziono {len(images)} obrazów do przetworzenia.")

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        ) as progress:
            task_transcribing = progress.add_task(
                "Transcribing", total=len(images))
            for idx, img_path in enumerate(images, 1):
                self.process_image(img_path, idx, len(images))
                progress.advance(task_transcribing)

    def process_image(self, img_path: Path, idx: int, total: int) -> None:
        self.log.info(f"Przetwarzam obraz {idx}/{total}: {img_path}")
        img_input_path_rel = img_path.relative_to(self.input)
        transcription_dir = self.transcription.joinpath(
            img_input_path_rel).parent
        transcription_path = transcription_dir.joinpath(img_path.stem + ".txt")
        transcription_dir.mkdir(parents=True, exist_ok=True)

        transcription_image_dir = self.transcription_images.joinpath(
            img_input_path_rel).parent
        transcription_image_path = transcription_image_dir.joinpath(
            img_path.name)
        transcription_image_dir.mkdir(parents=True, exist_ok=True)

        self.log.info(
            f'Processing {str(idx).zfill(len(str(total)))}: "{img_input_path_rel}"')

        if self.skip_existing and transcription_path.is_file():
            self.log.warning("Transcription already exists, skipping")
            return

        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            with torch.no_grad():
                results = self.model.predict_detections_and_associations([
                                                                         image_np])
                text_bboxes = [x["texts"] for x in results]
                ocr_results = self.model.predict_ocr([image_np], text_bboxes)

            transcript = self.model.generate_transcript_for_single_image(
                results[0], ocr_results[0])
            cleaned_transcript = self.clean_transcript(transcript)

            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_transcript)
            self.log.info(f"Transkrypcja zapisana w: {transcription_path}")

            self.model.visualise_single_image_prediction(
                image_np, results[0], filename=str(transcription_image_path))
            self.log.info(
                f"Wizualizacja zapisana w: {transcription_image_path}")

            self.log.info(
                f"Image successfully transcribed and saved as {transcription_path}")
        except Exception as e:
            self.log.error(
                f"An error occurred while processing the image: {str(e)}")
            self.log.exception("Szczegóły błędu:")
            self.log.error(
                f"Błąd podczas przetwarzania obrazu {img_path}: {str(e)}")

        self.log.info(f"Zakończono przetwarzanie obrazu: {img_path}")

    def clean_transcript(self, transcript: str) -> str:
        lines = transcript.split('\n')
        cleaned_lines = []
        for line in lines[1:]:  # Skip the first line (### Transcript ###)
            line = re.sub(r'<[^>]+>:\s*', '', line)  # Remove <???>: patterns
            if line.strip():
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)


app = typer.Typer()


@app.command()
def main(
    input: Path = typer.Option(
        Path("input"), "--input", "-i", help="Input folder"),
    transcription: Path = typer.Option(Path(
        "transcription"), "--transcription", "-t", help="Transcription output folder"),
    transcription_images: Path = typer.Option(Path(
        "transcription_images"), "--transcription-images", "-ti", help="Transcription images output folder"),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", "-se", help="Skip existing output files"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose mode"),
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.ERROR,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
    )

    transcriber = MangaTranscriber(
        input=input,
        transcription=transcription,
        transcription_images=transcription_images,
        skip_existing=skip_existing,
    )
    transcriber.run()


if __name__ == "__main__":
    app()
