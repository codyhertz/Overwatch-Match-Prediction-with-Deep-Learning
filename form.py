from flask_wtf import FlaskForm
from wtforms import SelectField

class Form(FlaskForm):
    map = SelectField('Map', choices=[('1', 'Blizzard World'),
                                      ('2', 'Busan City'),
                                      ('3', 'Dorado'),
                                      ('4', 'Eichenwalde'),
                                      ('6', 'Hanamura'),
                                      ('7', 'Hollywood'),
                                      ('8', 'Horizon Lunar Colony'),
                                      ('9', 'Illios'),
                                      ('10', 'Junkertown'),
                                      ('11', 'King\'s Row'),
                                      ('12', 'Lijiang Tower'),
                                      ('13', 'Nepal'),
                                      ('14', 'Numbani'),
                                      ('15', 'Oasis'),
                                      ('16', 'Rialto'),
                                      ('17', 'Route 66'),
                                      ('0', 'Temple of Anubis'),
                                      ('18', 'Volskaya Industries'),
                                      ('5', 'Watchpoint Gibraltar')
                                      ])

    position = SelectField('Position', choices=[])

    outcome = SelectField('Outcome', choices=[('0', 'Defeat'), ('1', 'Victory')])

    team_member1 = SelectField('Player\'s Team Member 1', choices=[('0', 'None'),
                                               ('24', 'Ana'),
                                               ('8', 'Ashe'),
                                               ('9', 'Bastion'),
                                               ('25', 'Brigitte'),
                                               ('1', 'D.Va'),
                                               ('10', 'Doomfist'),
                                               ('11', 'Genji'),
                                               ('12', 'Hanzo'),
                                               ('13', 'Junkrat'),
                                               ('26', 'Lucio'),
                                               ('14', 'McCree'),
                                               ('15', 'Mei'),
                                               ('27', 'Mercy'),
                                               ('28', 'Moira'),
                                               ('2', 'Orisa'),
                                               ('16', 'Pharah'),
                                               ('17', 'Reaper'),
                                               ('3', 'Reinhardt'),
                                               ('4', 'Roadhog'),
                                               ('18', 'Soldier: 76'),
                                               ('19', 'Sombra'),
                                               ('20', 'Symmetra'),
                                               ('21', 'Thorbjorn'),
                                               ('22', 'Tracer'),
                                               ('23', 'Widowmaker'),
                                               ('5', 'Winston'),
                                               ('6', 'Wrecking Ball'),
                                               ('7', 'Zarya'),
                                               ('29', 'Zenyatta')
                                               ])

    team_member2 = SelectField('Player\'s Team Member 2', choices=[])

    team_member3 = SelectField('Player\'s Team Member 3', choices=[])

    team_member4 = SelectField('Player\'s Team Member 4', choices=[])

    team_member5 = SelectField('Player\'s Team Member 5', choices=[])

    team_member6 = SelectField('Player\'s Team Member 6', choices=[])

    enemy_member1 = SelectField('Opposing Team Member 1', choices=[('0', 'None'),
                                               ('24', 'Ana'),
                                               ('8', 'Ashe'),
                                               ('9', 'Bastion'),
                                               ('25', 'Brigitte'),
                                               ('1', 'D.Va'),
                                               ('10', 'Doomfist'),
                                               ('11', 'Genji'),
                                               ('12', 'Hanzo'),
                                               ('13', 'Junkrat'),
                                               ('26', 'Lucio'),
                                               ('14', 'McCree'),
                                               ('15', 'Mei'),
                                               ('27', 'Mercy'),
                                               ('28', 'Moira'),
                                               ('2', 'Orisa'),
                                               ('16', 'Pharah'),
                                               ('17', 'Reaper'),
                                               ('3', 'Reinhardt'),
                                               ('4', 'Roadhog'),
                                               ('18', 'Soldier: 76'),
                                               ('19', 'Sombra'),
                                               ('20', 'Symmetra'),
                                               ('21', 'Thorbjorn'),
                                               ('22', 'Tracer'),
                                               ('23', 'Widowmaker'),
                                               ('5', 'Winston'),
                                               ('6', 'Wrecking Ball'),
                                               ('7', 'Zarya'),
                                               ('29', 'Zenyatta')
                                               ])

    enemy_member2 = SelectField('Opposing Team Member 2', choices=[])

    enemy_member3 = SelectField('Opposing Team Member 3', choices=[])

    enemy_member4 = SelectField('Opposing Team Member 4', choices=[])

    enemy_member5 = SelectField('Opposing Team Member 5', choices=[])

    enemy_member6 = SelectField('Opposing Team Member 6', choices=[])

    algorithm = SelectField('Model', choices=[('0', 'Deep Neural Network with Complex Data '),
                                              ('1', 'Deep Neural Network with Simplified Data'),
                                              ('2', 'Shallow Neural Network with Complex Data'),
                                              ('3', 'Shallow Neural Network with Simplifeid Data'),
                                              ('4', 'SVM with Complex Data'),
                                              ('5', 'SVM with Simplifeid Data'),
                                              ('6', 'Logistical Regression with Complex Data'),
                                              ('7', 'Logistical Regression with Simplified Data')
                                             ])
