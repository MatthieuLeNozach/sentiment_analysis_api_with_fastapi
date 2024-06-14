from project import create_app
import logging

logger = logging.getLogger(__name__)

app = create_app()
#celery = app.celery_app


