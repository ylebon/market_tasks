from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from core.task_step import TaskStep


class Task(TaskStep):
    """
    Select features importances

    """

    def run(self, features, target, limit=10, selectors=None):
        # add selectors
        if selectors is None:
            selectors = ["xgboost", "rfe", "kbest", "boruta"]
        self.log.info(f"msg='running features importances selector' selectors='{selectors}'")

        # X, y
        X = features.values
        y = target.values.ravel()

        # result
        result = dict()

        # Select N important features
        if "xgboost" in selectors:
            try:
                self.log.info("msg='running XGBoost features importances'")
                model = XGBClassifier()
                model.fit(X, y)
                model.get_booster().get_score(importance_type='weight')
                features_importances = list(zip(list(features), model.feature_importances_))
                features_importances = sorted(features_importances, key=lambda x: x[1], reverse=True)
                result['xgboost'] = features_importances
            except Exception as error:
                self.log.error(f"msg='failed to run xgboost features selection' error='{error}'")

        # Select K best
        if "kbest" in selectors:
            try:
                self.log.info("msg='running Kbest features importances'")
                k = limit or 10
                X_scaled = MinMaxScaler().fit_transform(X)
                features = SelectKBest(score_func=chi2, k=k).fit(X_scaled, y)
                features_importances = list(zip(list(X), features.scores_))
                features_importances = sorted(features_importances, key=lambda x: x[1], reverse=True)
                result['kbest'] = features_importances
            except Exception as error:
                self.log.error(f"msg='failed to run kbest features selection' error='{error}'")

        # Select RFE
        if "rfe" in selectors:
            try:
                self.log.info("msg='running RFE features importances'")
                model = LogisticRegression()
                rfe = RFE(model, limit).fit(X, y)
                features_importances = list(zip(list(X), rfe.ranking_))
                features_importances = sorted(features_importances, key=lambda x: x[1], reverse=False)
                result['rfe'] = features_importances
            except Exception as error:
                self.log.error(f"msg='failed to run RFE features selection' error='{error}'")

        # Boruta
        if "boruta" in selectors:
            try:
                self.log.info("msg='running Boruta features importances'")
                rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
                feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
                feat_selector.fit(X, y)
                features_importances = list(zip(list(X), feat_selector.ranking_))
                features_importances = sorted(features_importances, key=lambda x: x[1], reverse=False)
                result['boruta'] = features_importances
            except Exception as error:
                self.log.error(f"msg='failed to run Boruta features selection' error='{error}'")

        # Common
        return result


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    task = Task("sync_s3")
    task.run()
