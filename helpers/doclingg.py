from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

artifacts_path = "/home/yash/.cache/docling/models"
accel_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CPU)

table_options = TableStructureOptions(mode=TableFormerMode.FAST, do_cell_matching=False)
pipeline_options = ThreadedPdfPipelineOptions(
    do_ocr=False,
    do_table_structure=True,
    table_structure_options=table_options,
    images_scale=0.7,
    generate_page_images=False,
    generate_picture_images=False,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    accelerator_options=accel_options,
    artifacts_path=artifacts_path,
)
pdf_to_docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options, pdf_backend="pypdfium2"
        )
    }
)
