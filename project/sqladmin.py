from sqladmin import Admin, ModelView
from project.fu_core.users.models import User
from project.fu_core.users import models
# Create a ModelView for the User model
class UserAdmin(ModelView, model=User):
    column_list = [models.User.id, models.User.email, models.User.is_active, models.User.is_superuser]

