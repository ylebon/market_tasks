import uuid

from telegram import Bot

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """Return last sharpe ratio

    """

    bot = None
    app_id = None

    def run(self, message):
        """
        Send message

        """
        services_config = ConfigServices.create()
        token = services_config.get_value("TELEGRAM.TOKEN")
        chat_id = services_config.get_value("TELEGRAM.CHAT_ID")
        if not self.bot:
            self.bot = Bot(token=token)
        if not self.app_id:
            self.app_id = uuid.uuid4().hex[:10]
        self.bot.send_message(chat_id=chat_id, text=message)
