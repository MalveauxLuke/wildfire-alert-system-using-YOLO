import os
import smtplib
import logging
from email.message import EmailMessage
import requests
from datetime import datetime, timedelta
   
# Create logger for pipeline with given logging level
def setup_logger(level: str = "INFO"):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, level),
    )
    return logging.getLogger("pipeline")

logger = setup_logger()

# Tracks the time of the last alert
_last_alert_time = datetime.min

# Checks whether an alert can be sent based on specified cooldown
def can_alert(cooldown_seconds):
    global _last_alert_time
    now = datetime.utcnow()
    if (now - _last_alert_time) > timedelta(seconds=cooldown_seconds):
        _last_alert_time = now
        return True
    return False

# Sends an email with a text message and an attached image
def send_email(cfg, subject, body, image_path):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["sender"]
    msg["To"] = cfg["recipient"]
    msg.set_content(body)

    with open(image_path, "rb") as img:
        img_data = img.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=image_path)
        
    with smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"]) as server:
        server.starttls()
        server.login(cfg["sender"], os.getenv(cfg["password_env"]))
        server.send_message(msg)
    logger.info("Potential wildfire detected. Email alert sent.")