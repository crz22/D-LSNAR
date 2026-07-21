/****************************************************************************
** Meta object code from reading C++ file 'dialog.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/cpp/dialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'dialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_Reconstruction_Dialog[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      23,   22,   22,   22, 0x08,
      43,   22,   22,   22, 0x08,
      64,   22,   22,   22, 0x08,
      83,   22,   22,   22, 0x08,
     105,   22,   22,   22, 0x08,
     125,   22,   22,   22, 0x08,
     146,   22,   22,   22, 0x08,
     165,   22,   22,   22, 0x08,
     187,   22,   22,   22, 0x08,
     215,   22,   22,   22, 0x08,
     243,   22,   22,   22, 0x08,
     271,   22,   22,   22, 0x08,
     301,   22,   22,   22, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_Reconstruction_Dialog[] = {
    "Reconstruction_Dialog\0\0Select_input_path()\0"
    "Select_marker_path()\0Select_save_path()\0"
    "Select_pytorch_path()\0update_input_path()\0"
    "update_marker_path()\0update_save_path()\0"
    "update_pytorch_path()\0Select_configuration_path()\0"
    "update_configuration_path()\0"
    "onConfigPathButtonClicked()\0"
    "onConfigPathEditingFinished()\0"
    "Start_neuron_reconstruction()\0"
};

void Reconstruction_Dialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        Reconstruction_Dialog *_t = static_cast<Reconstruction_Dialog *>(_o);
        switch (_id) {
        case 0: _t->Select_input_path(); break;
        case 1: _t->Select_marker_path(); break;
        case 2: _t->Select_save_path(); break;
        case 3: _t->Select_pytorch_path(); break;
        case 4: _t->update_input_path(); break;
        case 5: _t->update_marker_path(); break;
        case 6: _t->update_save_path(); break;
        case 7: _t->update_pytorch_path(); break;
        case 8: _t->Select_configuration_path(); break;
        case 9: _t->update_configuration_path(); break;
        case 10: _t->onConfigPathButtonClicked(); break;
        case 11: _t->onConfigPathEditingFinished(); break;
        case 12: _t->Start_neuron_reconstruction(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData Reconstruction_Dialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject Reconstruction_Dialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_Reconstruction_Dialog,
      qt_meta_data_Reconstruction_Dialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Reconstruction_Dialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Reconstruction_Dialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Reconstruction_Dialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Reconstruction_Dialog))
        return static_cast<void*>(const_cast< Reconstruction_Dialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int Reconstruction_Dialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
